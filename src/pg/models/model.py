# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/models/model.py
import csv
import logging
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.pg.utils.config import CONFIG
from src.pg.utils.metrics import MetricsEvaluator
import pandas as pd
import wandb

# Initialize a logger for debugging and output control
logger = logging.getLogger(__name__)


class FlanT5FineTuner(pl.LightningModule):

    def __init__(self, model_name, model_dir, file_label=""):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()
        # Save only essential hyperparameters
        self.save_hyperparameters('model_name')

        # Store model_dir and file_label as instance variables
        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # Load T5 model and tokenizer with configurations specified in CONFIG
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Set unique file paths using `file_label` to prevent overwriting
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Initialize buffers for validation
        self.epoch_validation_details = []  # Storage for each validation epoch
        self.epoch_scores = []  # Validation scores buffer

        # Initialize buffers for testing
        self.epoch_test_details = []  # Storage for each test epoch
        self.epoch_test_scores = []  # Test scores buffer

        # Initialize MetricsEvaluator to handle custom scoring for rewards
        self.metrics_evaluator = MetricsEvaluator()


    def forward(self, input_ids, attention_mask):
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG['max_gen_length'],
            do_sample=True, 
            temperature=1.5,
            output_scores=True,
            return_dict_in_generate=True
        )
        generated_tokens = outputs.sequences
        logits = outputs.scores
        return generated_tokens, logits

    def apply_vocab_masking(self, logits):
        """
        Masks logits for tokens beyond the vocabulary size of the tokenizer.
        Handles both 2D and 3D tensors for compatibility with generated logits.
        """
        vocab_size = self.tokenizer.vocab_size

        # Check if logits is 2D (batch_size, vocab_size) or 3D (batch_size, sequence_length, vocab_size)
        if logits.dim() == 2:
            # Mask for 2D logits (each decoding step in generate)
            masked_logits = logits.clone()
            masked_logits[:, vocab_size:] = -float('inf')
        elif logits.dim() == 3:
            # Mask for 3D logits (entire sequence logits from forward pass)
            masked_logits = logits.clone()
            masked_logits[:, :, vocab_size:] = -float('inf')
        else:
            raise ValueError(f"Unexpected logits dimension: expected 2 or 3, got {logits.dim()}")

        return masked_logits


    def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards, baseline):
        """
        Calculates policy gradient loss based on generated tokens and rewards.
        Handles the case where BART scores are negative by flipping the sign of rewards.
        """
        # Stack logits along the sequence dimension and apply log softmax
        logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
        logits = self.apply_vocab_masking(logits)  # Apply masking to stacked logits

        # Detach logits after applying masking to free memory
        logits = logits.detach()

        # Gather the log probabilities for the generated tokens
        labels_for_indexing = generated_tokens[:, 1:].contiguous()
        token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

        # Create a mask to ignore padding tokens
        padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
        token_log_probs = token_log_probs * padding_mask.float()

        # Sum log probabilities across the sequence dimension
        sequence_log_prob_sum = token_log_probs.sum(dim=1)

        # Handle special case for BART (negative rewards)
        if CONFIG.get("reward_metric") == "bart":
           rewards = rewards + 4  # add a Baseline to move to the positif size but you keep the magnitude
            # Calculate policy gradient loss
        return -(rewards * sequence_log_prob_sum).mean()


    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # Forward pass
        generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Detach logits immediately after forward pass to free memory
        logits = [logit.detach() for logit in logits]

        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Get ground-truth references
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Calculate rewards
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()
            rewards = rewards - dynamic_baseline

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

            # Calculate PG loss
        pg_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=dynamic_baseline)

        # Logging
        self.log('training_pg_loss', pg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('training_pg_reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('training_pg_baseline', dynamic_baseline, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.log('training_pg_delta_m1_mean', delta_m1.mean().item(), on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

        logger.info(
            f'[TRAIN] PG Loss: {pg_loss}, Baseline: {dynamic_baseline}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_loss


    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        print(f"Validation Step: Processing batch {batch_idx}")
        generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        
        # Detach logits to prevent memory overflow
        logits = [logit.detach() for logit in logits]

        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Calculate scores
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Handle the different experiments
        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

        # Compute PG validation loss (baseline = 0.0, since no updates occur)
        pg_val_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=0.0)

        # Log validation metrics
        self.log('validation_pg_loss', pg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_pg_baseline', dynamic_baseline, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.log('validation_pg_delta_m1_mean', delta_m1.mean().item(), on_epoch=True, prog_bar=True, logger=True)

        # Save validation details
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        logger.info(
            f'[VALIDATION] Epoch {self.current_epoch} | PG Loss: {pg_val_loss}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_val_loss


    def on_validation_epoch_end(self):
        """
        Finalize and save validation results at the end of the validation epoch.
        """
        print("Validation Epoch End")
        if self.epoch_validation_details:
            print(f"Saving {len(self.epoch_validation_details)} validation details to {self.val_csv_file_path}.")
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)

        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            print(f"Overall validation score: {overall_val_score}")
            self.log("overall_score", overall_val_score, prog_bar=True, logger=True)

        # Clear buffers for next validation run
        self.epoch_validation_details.clear()
        self.epoch_scores.clear()



    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Detach logits
        logits = [logit.detach() for logit in logits]

        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Compute scores
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Handle the different experiments
        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

        # Compute PG test loss (baseline = 0.0, since no updates occur)
        pg_test_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=0.0)

        # Log test metrics
        self.log('test_pg_loss', pg_test_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pg_baseline', dynamic_baseline, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.log('test_pg_delta_m1_mean', delta_m1.mean().item(), on_epoch=True, prog_bar=True, logger=True)

        # Save test details
        for i in range(len(generated_texts)):
            self.epoch_test_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        logger.info(
            f'[TEST] Epoch {self.current_epoch} | PG Loss: {pg_test_loss}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_test_loss

  

    def on_test_epoch_end(self):
        """
        Finalize and save test results at the end of the test epoch.
        """
        print("Test Epoch End")
        if self.epoch_test_details:
            print(f"Saving {len(self.epoch_test_details)} test details to {self.test_csv_file_path}.")
            self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)

        if self.epoch_test_scores:
            overall_test_score = torch.tensor(self.epoch_test_scores).mean().item()
            print(f"Overall test score: {overall_test_score}")
            self.log("test_overall_score", overall_test_score, prog_bar=True, logger=True)

        # Clear buffers for next test run
        self.epoch_test_details.clear()
        self.epoch_test_scores.clear()

    def log_to_csv(self, csv_file_path, details, epoch=None):
        """
        Writes the details to the specified CSV file.
        """
        print(f"Writing {len(details)} entries to {csv_file_path}.")
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()
            for detail in details:
                detail['Epoch'] = epoch
            writer.writerows(details)

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
