# src/pg/models/model_dpo.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import pandas as pd
import Levenshtein
from src.dpo.utils.config_dpo import CONFIG
from src.dpo.utils.metrics_dpo import MetricsEvaluator
import logging

logger = logging.getLogger(__name__)

class FlanT5DPOTrainer(pl.LightningModule):

    def __init__(self, model_name: str, model_dir: Path, file_label: str = ""):
        super().__init__()
        self.save_hyperparameters()

        # 1) Load models + tokenizer
        self.model     = T5ForConditionalGeneration.from_pretrained(model_name)
        self.ref_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # 2) Freeze reference
        for p in self.ref_model.parameters():
            p.requires_grad = False

        # 3) DPO hyperparams
        self.beta      = CONFIG["dpo"]["beta"]
        self.loss_type = CONFIG["dpo"]["loss_type"]

        # 4) Metrics & CSV paths
        self.metrics_evaluator    = MetricsEvaluator()
        self.val_csv_path         = Path(model_dir) / f"dpo_validation{file_label}.csv"
        self.test_csv_path        = Path(model_dir) / f"dpo_test{file_label}.csv"

        # 5) Buffers for generations
        self.epoch_validation_details = []
        self.epoch_test_details       = []

    def _get_beta(self, step: int) -> float:
        sched   = CONFIG["dpo"]["beta_schedule"]
        init, fin, total = sched["initial"], sched["final"], sched["num_steps"]
        if step >= total:
            return fin
        return init + (fin - init) * (step / total)

    def forward(self, input_ids, attention_mask):
        with torch.amp.autocast("cuda", enabled=False):
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG["max_gen_length"],
                do_sample=True,
                temperature=0.7,
                top_k=30,
                top_p=0.8,
            )

    def training_step(self, batch, batch_idx):
        bs = batch["chosen_input_ids"].size(0)
        for k in ["chosen_input_ids", "chosen_attention_mask", "chosen_labels",
                  "rejected_input_ids", "rejected_attention_mask", "rejected_labels"]:
            batch[k] = batch[k].to(self.device)

        loss = self.compute_dpo_loss(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        return loss

    def _get_batch_logps(self, input_ids, attention_mask, labels, model=None):
        model = model or self.model
        input_ids      = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels         = labels.to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        logps = F.log_softmax(out.logits, dim=-1)

        pad_id     = self.tokenizer.pad_token_id
        valid_mask = (labels != pad_id) & (labels != -100)

        safe_labels = labels.clone()
        safe_labels[~valid_mask] = pad_id

        token_logps = torch.gather(logps,
                                   dim=2,
                                   index=safe_labels.unsqueeze(-1)
                                  ).squeeze(-1)
        token_logps = token_logps * valid_mask
        return token_logps.sum(dim=-1)

    def compute_dpo_loss(self, batch):
        p_chosen = self._get_batch_logps(
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        p_reject = self._get_batch_logps(
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

        with torch.no_grad():
            r_chosen = self._get_batch_logps(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
                model=self.ref_model,
            )
            r_reject = self._get_batch_logps(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
                model=self.ref_model,
            )

        kl = (p_chosen.exp() * (p_chosen - r_chosen)).mean()
        self.log("train/kl_divergence", kl, prog_bar=True, batch_size=p_chosen.size(0))

        self.beta = self._get_beta(self.global_step)

        from trl.trainer.dpo_trainer import DPOTrainer
        from trl.trainer.dpo_config  import FDivergenceType
        from types import SimpleNamespace

        dummy = object.__new__(DPOTrainer)
        dummy.beta              = self.beta
        dummy.loss_type         = self.loss_type
        dummy.label_smoothing   = CONFIG["dpo"].get("label_smoothing", 0.0)
        dummy.reference_free    = CONFIG["dpo"].get("reference_free", False)
        dummy.f_divergence_type = CONFIG["dpo"].get("f_divergence_type", FDivergenceType.REVERSE_KL)
        dummy.accelerator       = SimpleNamespace(device=self.device)

        losses, r_chosen_reward, r_reject_reward = DPOTrainer.dpo_loss(
            dummy, p_chosen, p_reject, r_chosen, r_reject
        )

        self.log_dict({
            "train/chosen_rewards":  r_chosen_reward.mean(),
            "train/rejected_rewards": r_reject_reward.mean(),
            "train/beta":            torch.tensor(self.beta),
        }, prog_bar=True, batch_size=losses.size(0))

        return losses.mean()

    def validation_step(self, batch, batch_idx):
        batch["input_ids"]      = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)
        bs = batch["input_ids"].size(0)

        generated = self(batch["input_ids"], batch["attention_mask"])
        gen_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        sim_edited   = self.metrics_evaluator.calculate_score(
            gen_texts, [str(e) for e in batch["edited_ending"]]
        )
        sim_original = self.metrics_evaluator.calculate_score(
            gen_texts, [str(o) for o in batch["original_ending"]]
        )
        edit_dist = torch.tensor([
            Levenshtein.distance(g, o) / max(len(g), len(o))
            for g, o in zip(gen_texts, batch["original_ending"])
        ], device=self.device)

        self.log_dict({
            "val/alignment":     (sim_edited > sim_original).float().mean(),
            "val/sim_edited":    sim_edited.mean(),
            "val/sim_original":  sim_original.mean(),
            "val/delta_sim":     (sim_edited - sim_original).mean(),
            "val/edit_distance": edit_dist.mean(),
            "val/length_ratio":  torch.tensor([
                len(g)/len(o) for g, o in zip(gen_texts, batch["original_ending"])
            ], device=self.device).mean(),
        }, batch_size=bs)

        self._log_generation_details(batch, gen_texts, validation=True)

    def test_step(self, batch, batch_idx):
        """
        Mirror validation_step but write to test CSV instead.
        """
        batch["input_ids"]      = batch["input_ids"].to(self.device)
        batch["attention_mask"] = batch["attention_mask"].to(self.device)

        generated = self(batch["input_ids"], batch["attention_mask"])
        gen_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # We skip metric logging here, but you could mirror validation metrics if desired.
        self._log_generation_details(batch, gen_texts, validation=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])

    def _log_generation_details(self, batch, generated_texts, validation=True):
        csv_path = self.val_csv_path if validation else self.test_csv_path
        rows = []
        for i, gen in enumerate(generated_texts):
            rows.append({
                "premise":          batch["premise"][i],
                "initial":          batch["initial"][i],
                "counterfactual":   batch["counterfactual"][i],
                "original_ending":  batch["original_ending"][i],
                "edited_ending":    batch["edited_ending"][i],
                "generated_ending": gen,
            })
        df = pd.DataFrame(rows)
        df.to_csv(
            csv_path,
            mode="a" if csv_path.exists() else "w",
            header=not csv_path.exists(),
            index=False
        )
