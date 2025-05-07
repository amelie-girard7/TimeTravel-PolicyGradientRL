# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/ppo/models/model_ppo.py
import gc
import logging
import os
import traceback

import torch
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from transformers.models.t5.configuration_t5 import T5Config as SafeT5Config
import pytorch_lightning as pl
from pathlib import Path
from src.ppo.utils.config_ppo import CONFIG
from src.ppo.utils.metrics import MetricsEvaluator
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.DEBUG)

# Allow T5Config for safe loading
add_safe_globals([SafeT5Config])

logger = logging.getLogger(__name__)

class FlanT5PPOFineTuner(pl.LightningModule):

    def __init__(self, model_name, model_dir, file_label=""):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.automatic_optimization = False

        self.model_dir = Path(model_dir)
        self.file_label = file_label
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()
        logger.info(f"Loaded pretrained T5 model from '{model_name}'")
        logger.info(f"Model config: d_model={self.model.config.d_model}, "
                    f"num_layers={self.model.config.num_layers}, "
                    f"vocab_size={self.model.config.vocab_size}")

        # Replace tokenizer initialization with this:
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            legacy=False,
            model_max_length=CONFIG['max_length'],
            truncation_side='left',
            padding_side='right',
            additional_special_tokens=[],
            clean_up_tokenization_spaces=True  # new!
        )

        self._validate_tokenizer()
        logger.info(f"Tokenizer initialized with vocab_size={self.tokenizer.vocab_size}")

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.d_model, CONFIG.get("value_head_hidden_size", 512)),
            torch.nn.ReLU(),
            torch.nn.Linear(CONFIG.get("value_head_hidden_size", 512), 1)
        )

        if CONFIG.get("ppo_from_checkpoint", False) and CONFIG["ppo_checkpoint_path"]:
            self._load_checkpoint_safely(CONFIG["ppo_checkpoint_path"])
        else:
            logger.info("Training value head from scratch.")
            self._initialize_value_head()

        self.verify_initialization()  # Add this line

        self.trajectory_buffer = []

        self.clip_epsilon = CONFIG["ppo_clip_epsilon"]
        self.ppo_epochs = CONFIG["ppo_epochs"]
        self.entropy_coef = CONFIG["entropy_coef"]
        self.max_trajectory_length = CONFIG["max_trajectory_length"]
        self.gamma = CONFIG["gamma"]
        self.lam = CONFIG["lambda"]
        self.value_coef = CONFIG.get("value_coef", 0.5)

        self.buffer_token_count = 0

        self.metrics_evaluator = MetricsEvaluator()
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"
        self.epoch_validation_details = []
        self.epoch_test_details = []

        self.episode_rewards = []
        self.episode_lengths = []
        self.value_estimates = []
        self.kl_divergences = []
        self.reward_components = {  
            'edited_similarity': [],
            'original_similarity': [],
            'contrastive': []
        }

    def _validate_tokenizer(self):
        """T5-compatible tokenizer validation"""
        # 1. Check essential tokens exist
        required_tokens = {
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'unk_token': self.tokenizer.unk_token
        }
        for name, token in required_tokens.items():
            if token is None:
                raise ValueError(f"Tokenizer missing required {name}")

        # 2. Test encode/decode with T5 expectations
        test_samples = [
            ("Basic text", "Basic text"),  # No longer expecting </s> suffix
            ("Numbers 123", "Numbers 123"),
            ("Punctuation!?", "Punctuation!?")
        ]

        for original, expected in test_samples:
            # Encode with add_special_tokens=False
            encoded = self.tokenizer.encode(original, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
            
            # Normalize whitespace for comparison
            decoded_clean = ' '.join(decoded.split())
            expected_clean = ' '.join(expected.split())
            
            if decoded_clean != expected_clean:
                logger.error(
                    f"Tokenizer mismatch:\n"
                    f"Original: '{original}'\n"
                    f"Expected: '{expected_clean}'\n"
                    f"Got:      '{decoded_clean}'\n"
                    f"Encoded: {encoded}"
                )
                raise ValueError("Tokenizer validation failed")

        logger.info(
            f" Tokenizer validated (vocab_size={self.tokenizer.vocab_size})\n"
            f"Special tokens:\n"
            f"- pad_token='{self.tokenizer.pad_token}' (id={self.tokenizer.pad_token_id})\n"
            f"- eos_token='{self.tokenizer.eos_token}' (id={self.tokenizer.eos_token_id})\n"
            f"- unk_token='{self.tokenizer.unk_token}' (id={self.tokenizer.unk_token_id})"
        )

    def _validate_input_ids(self, input_ids):
        """Validate token IDs before processing"""
        vocab_size = self.tokenizer.vocab_size
        pad_id = self.tokenizer.pad_token_id

        if not isinstance(input_ids, torch.Tensor):
            logger.error(f"Input IDs must be tensor, got {type(input_ids)}")
            raise TypeError("Input IDs must be tensor")

        if (input_ids < 0).any() or (input_ids >= vocab_size).any():
            invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
            invalid_ids = input_ids[invalid_mask].unique().tolist()
            invalid_count = invalid_mask.sum().item()
            invalid_positions = torch.where(invalid_mask)[0].tolist()[:5]

            logger.error(
                f"Invalid token IDs detected:\n"
                f"- Total invalid: {invalid_count}/{input_ids.numel()}\n"
                f"- Invalid IDs: {invalid_ids}\n"
                f"- Sample positions: {invalid_positions}\n"
                f"- Vocab range: 0-{vocab_size-1}\n"
                f"- Input IDs range: {input_ids.min().item()}-{input_ids.max().item()}"
            )

            sample_bad = input_ids[invalid_mask][:3].tolist()
            sample_good = input_ids[~invalid_mask][:3].tolist()
            logger.info(
                f"Sample bad tokens: {sample_bad}\n"
                f"Sample good tokens: {sample_good}"
            )

            raise ValueError(f"Invalid token IDs: {invalid_ids[:10]}... (count={invalid_count})")

        logger.debug(f"Input IDs validated (shape={input_ids.shape}, "
                     f"range=[{input_ids.min().item()}, {input_ids.max().item()}])")       

    def _compare_texts(self, text1, text2):
        """T5-aware text comparison that handles special tokens"""
        def clean_text(text):
            if not isinstance(text, str):
                text = str(text)
            return text.replace(self.tokenizer.eos_token, '').strip()

        return clean_text(text1) == clean_text(text2)

    def prepare_ppo_batch(self, trajectory_buffer):
        """Handle variable-length sequences with detailed debugging"""

        """Add input validation"""
        assert len(trajectory_buffer) > 0, "Empty trajectory buffer"
        for item in trajectory_buffer:
            assert 'input_ids' in item, "Missing input_ids in trajectory"
            assert 'generated_tokens' in item, "Missing generated_tokens in trajectory"

        if not trajectory_buffer:
            raise ValueError("Empty trajectory buffer")
        
        batch = {}
        
        # Enhanced sequence padding with debugging
        for key in ['input_ids', 'generated_tokens']:
            sequences = [item[key] for item in trajectory_buffer]

            logger.debug(f"Padding {key} with {len(sequences)} sequences")
            for i, seq in enumerate(sequences):
                logger.debug(f"Sequence {i} length: {len(seq)}")

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                sequences,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
        )

        
        # Scalar values
        for key in ['old_log_probs', 'rewards', 'values']:
            values = [item[key] for item in trajectory_buffer]
            logger.debug(f"Stacking {key} with shapes: {[v.shape for v in values]}")
            batch[key] = torch.stack(values)
        
        # Final validation
        logger.debug("Final batch shapes:")
        for k, v in batch.items():
            logger.debug(f"{k}: {v.shape}")
        
        return batch
   
    def get_value(self, input_ids, generated_tokens, chunk_size: int = 4):
        """
        Computes value predictions based on decoder hidden states.
        Uses the mean of decoder hidden states as input to value head.
        """
        # Log entry shapes
        logger.debug(
            f"[get_value] Called with input_ids.shape={input_ids.shape}, "
            f"generated_tokens.shape={generated_tokens.shape}"
        )
        with torch.no_grad():
            batch_size = input_ids.size(0)
            outputs = []

            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                ids = input_ids[start:end]
                gen = generated_tokens[start:end]
                logger.debug(
                    f"[get_value] Chunk {start}:{end} — ids.shape={ids.shape}, gen.shape={gen.shape}"
                )

                # Prepare decoder inputs and mask
                decoder_input_ids = gen[:, :-1]
                attention_mask = (ids != self.tokenizer.pad_token_id).long()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = self.model(
                        input_ids=ids,
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                # Grab last hidden state [B, L, D]
                last_hidden = out.decoder_hidden_states[-1]
                logger.debug(
                    f"[get_value] last_hidden.shape={last_hidden.shape}"
                )

                # Mean-pool over sequence length → [B, D]
                mean_hidden = last_hidden.mean(dim=1)
                logger.debug(
                    f"[get_value] mean_hidden.shape={mean_hidden.shape}"
                )

                # Value head → [B, 1]
                values = self.value_head(mean_hidden)
                # Sanitize
                values = torch.nan_to_num(values, nan=0.0, posinf=1e4, neginf=-1e4)
                logger.debug(
                    f"[get_value] chunk values (pre-squeeze) shape={values.shape}, "
                    f"mean={values.mean().item():.4f}, std={values.std().item():.4f}"
                )

                outputs.append(values)

            # Concatenate all chunks → [batch_size, 1], then squeeze → [batch_size]
            all_values = torch.cat(outputs, dim=0).squeeze(-1)
            logger.debug(
                f"[get_value] Returning all_values.shape={all_values.shape}, "
                f"sample_values={all_values[:3].tolist()}"
            )

            # mirror to wandb
            self.log('value_head/mean', values.mean(), on_step=False, on_epoch=True)
            self.log('value_head/std',  values.std(),  on_step=False, on_epoch=True)

            return all_values

    def calculate_gae(self, rewards, values, done_mask):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0

        # Handle sequence boundaries properly
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1 or done_mask[t]:
                next_value = 0.0
                last_advantage = 0.0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_advantage = delta + self.gamma * self.lam * last_advantage
            advantages[t] = last_advantage

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def calculate_ppo_loss(self, old_log_probs, new_log_probs, advantages, rewards, values, entropy):
        """
        Computes the combined PPO loss:
        - Clipped policy gradient loss
        - Value function loss
        - Entropy bonus
        Includes detailed debug logs.
        """

        # --- Shape checks ---
        assert old_log_probs.dim() == 1, f"old_log_probs should be 1D [batch], got {old_log_probs.shape}"
        assert new_log_probs.dim() == 1, f"new_log_probs should be 1D [batch], got {new_log_probs.shape}"
        assert advantages.dim() == 1, f"advantages should be 1D [batch], got {advantages.shape}"
        assert rewards.dim() == 1, f"rewards should be 1D [batch], got {rewards.shape}"
        assert values.dim() == 1, f"values should be 1D [batch], got {values.shape}"
        assert entropy.dim() == 1, f"entropy should be 1D [batch], got {entropy.shape}"

        # --- NaN / Inf guards ---
        assert not torch.isnan(old_log_probs).any(), "NaN in old_log_probs"
        assert not torch.isnan(new_log_probs).any(), "NaN in new_log_probs"

        # --- Clamp advantages ---
        advantages = torch.clamp(advantages, -5.0, 5.0)

        # --- DEBUG PRINTS ---
        logger.info(f"[DEBUG] old_log_probs: mean={old_log_probs.mean().item():.4f}, std={old_log_probs.std().item():.4f}")
        logger.info(f"[DEBUG] new_log_probs: mean={new_log_probs.mean().item():.4f}, std={new_log_probs.std().item():.4f}")
        logger.info(f"[DEBUG] advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}")
        logger.info(f"[DEBUG] rewards: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}")
        logger.info(f"[DEBUG] values: mean={values.mean().item():.4f}, std={values.std().item():.4f}")
        logger.info(f"[DEBUG] entropy: mean={entropy.mean().item():.4f}, std={entropy.std().item():.4f}")

        # --- PPO Ratio ---
        log_ratio = new_log_probs - old_log_probs.detach()
        ratio = torch.exp(torch.clamp(log_ratio, -2.0, 2.0))

        # --- Policy loss ---
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # --- Value loss ---
        value_loss = F.mse_loss(values, rewards)

        # --- Entropy bonus ---
        entropy_loss = entropy.mean()

        # --- Total loss ---
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy_loss
        )

        # --- Extra metrics ---
        with torch.no_grad():
            kl_div = (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(-1).mean()
            self.kl_divergences.append(kl_div.item())
            value_error = (values - rewards).abs().mean()
            self.value_estimates.append(value_error.item())

            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
            self.log('ppo/clip_fraction', clip_fraction.item())

        return loss, policy_loss, value_loss

    def apply_vocab_masking(self, logits):
        vocab_size = self.tokenizer.vocab_size
        masked_logits = logits.clone()
        
        # Clip extreme values first
        masked_logits = torch.clamp(masked_logits, -50, 50)
        
        # Then apply masking
        if masked_logits.dim() == 2:
            masked_logits[:, vocab_size:] = -float('inf')
        elif masked_logits.dim() == 3:
            masked_logits[:, :, vocab_size:] = -float('inf')
        else:
            raise ValueError(f"Unexpected logits dimension: {masked_logits.dim()}")
        
        return masked_logits
 
    def calculate_rewards(self, generated_texts, edited_endings, original_endings):
        """
        Computes a positive PPO‐friendly reward signal.

        Raw reward:
            r_i = sim(gen_i, edit_i) + (sim(gen_i, edit_i) - sim(gen_i, orig_i))

        Then:
        1) subtract the batch mean → zero‐center
        2) shift so min > 0
        3) (optional) scale to [0,1]
        """
        # 1) Clean texts (remove EOS token, whitespace)
        def clean(t):
            return t.replace(self.tokenizer.eos_token, '').strip() if isinstance(t, str) else str(t)

        gen_clean = [clean(t) for t in generated_texts]
        edit_clean = [clean(t) for t in edited_endings]
        orig_clean = [clean(t) for t in original_endings]

        # 2) Compute similarity scores (detached from graph)
        sim_edited   = self.metrics_evaluator.calculate_score(gen_clean, edit_clean).detach()
        sim_original = self.metrics_evaluator.calculate_score(gen_clean, orig_clean).detach()

        # 3) Raw reward formula
        if CONFIG["ppo_experiment"] == "delta_m1":
            raw = sim_edited + (sim_edited - sim_original)
        else:
            raw = sim_edited

        # 4) Baseline subtraction → zero‐center
        baseline = raw.mean()
        centered = raw - baseline

        # 5) Shift so minimum reward > 0
        min_val = centered.min()
        # if min_val >= 0, add a small epsilon; else offset by -min_val + epsilon
        epsilon = 1e-3
        offset  = epsilon if min_val >= 0 else (-min_val + epsilon)
        positive = centered + offset

        # 6) Optional normalization to [0,1]
        max_val    = positive.max()
        normalized = positive / (max_val + 1e-8)

        # 7) Log intermediate stats for debugging
        self.log_dict({
            'train/sim_edited_mean':   sim_edited.mean(),
            'train/sim_original_mean': sim_original.mean(),
            'train/raw_reward_mean':   raw.mean(),
            'train/centered_min':      float(centered.min()),
            'train/offset':            offset,
            'train/positive_max':      float(positive.max()),
            'train/final_reward_mean': normalized.mean()
        }, prog_bar=True)

        # 8) Sanity check for NaNs/Infs
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            logger.error(
                "[REWARD DEBUG] Invalid values after normalization:\n"
                f"- NaNs: {(torch.isnan(normalized)).sum().item()}\n"
                f"- Infs: {(torch.isinf(normalized)).sum().item()}\n"
                f"- Sample: {normalized[:5]}"
            )
            raise ValueError("NaN/Inf detected in rewards")

        return normalized


    def validation_step(self, batch, batch_idx):
        """
        1) Pure inference via forward(…)
        2) Decode & optionally log sample outputs
        3) Compute reward metric
        4) Log avg/std/min/max with batch_size
        5) Record per‐example details
        """
        batch_size = batch["input_ids"].size(0)
        input_ids = batch['input_ids']

        # 1) generate
        generated_tokens = self.forward(input_ids)  # [B, L_out]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 2) log a couple of samples for debugging
        if batch_idx % CONFIG.get("log_samples_every_n_steps", 100) == 0:
            for i, txt in enumerate(generated_texts[:2]):
                logger.info(f"[Validation Batch {batch_idx}] Sample {i}: {txt}")

        # Get edited endings from batch
        edited_endings = [str(x) for x in batch['edited_ending']]
        
        # 5) Record per‐example details
        for i in range(len(generated_texts)):
            detail = {
                'Premise':          batch['premise'][i],
                'Initial':          batch['initial'][i],
                'Counterfactual':   batch['counterfactual'][i],
                'Original Ending':  batch['original_ending'][i],
                'Edited Ending':    edited_endings[i],
                'Generated Text':   generated_texts[i]
            }
            self.epoch_validation_details.append(detail)

        # 3) compute reward
        rewards = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)

        avg_reward = rewards.mean()
        std_reward = rewards.std()
        min_reward = rewards.min()
        max_reward = rewards.max()

        # 4) log all stats
        self.log('val/avg_reward', avg_reward, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/reward_std', std_reward, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/reward_min', min_reward, on_epoch=True, batch_size=batch_size)
        self.log('val/reward_max', max_reward, on_epoch=True, batch_size=batch_size)

    def test_step(self, batch, batch_idx):
        """
        Test step:
          - Generates tokens.
          - Decodes outputs and saves details.
          - Logs average reward.
        """
        input_ids = batch['input_ids']
        generated_tokens = self.forward(input_ids)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for i in range(len(generated_texts)):
            self.epoch_test_details.append({
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': batch['edited_ending'][i],
                'Generated Text': generated_texts[i]
            })
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        rewards = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)
        self.log('test/avg_reward', rewards.mean(), on_epoch=True)

    def update_ppo(self):
        """
        PPO update:
        1) prepare & flatten CPU‑side buffer
        2) move everything to GPU
        3) compute GAE advantages
        4) inner PPO loop (new log‑probs, loss)
        """
        # 1) Check for empty buffer
        if not self.trajectory_buffer:
            logger.warning("Skipping PPO update – trajectory_buffer is empty")
            return torch.tensor(0.0, device=self.device)

        # 2) Prepare big batch on CPU
        logger.info(f"Preparing PPO batch from {len(self.trajectory_buffer)} trajectories")
        batch = self.prepare_ppo_batch(self.trajectory_buffer)
        for key, val in batch.items():
            if key in ["input_ids", "generated_tokens"]:
                batch[key] = val.view(-1, val.size(-1))
            else:
                batch[key] = val.view(-1)
        logger.debug(f"After flattening, batch shapes: { {k: v.shape for k, v in batch.items()} }")

        # 3) Move to GPU
        for k in batch:
            batch[k] = batch[k].to(self.device)
        logger.debug("Moved entire PPO batch to GPU")

        # 4) Bootstrap values & compute GAE
        rewards = batch['rewards']
        values = batch['values']
        # append zero for bootstrap
        values = torch.cat([values, torch.zeros(1, device=self.device)], dim=0)
        done_mask = torch.zeros_like(rewards, device=self.device)
        logger.info(f"Computing GAE (γ={self.gamma}, λ={self.lam})")
        advantages = self.calculate_gae(rewards, values, done_mask)
        batch['advantages'] = advantages
        logger.debug(f"Advantages stats: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}")

        # advantage signal
        self.log('train/adv_mean', advantages.mean(), on_step=True, on_epoch=True)
        self.log('train/adv_std',  advantages.std(),  on_step=True)

        # 5) Inner PPO loop
        total_loss = torch.tensor(0.0, device=self.device)
        for epoch in range(self.ppo_epochs):
            logger.info(f"[PPO] Starting epoch {epoch+1}/{self.ppo_epochs}")

            # compute new log‑probs and entropy
            new_lp, entropy = self.compute_log_probs_and_entropy(
                batch['input_ids'],
                batch['generated_tokens']
            )
            # value estimates
            vals = self.get_value(
                input_ids=batch['input_ids'],
                generated_tokens=batch['generated_tokens']
            )

            # compute losses
            loss, policy_loss, value_loss = self.calculate_ppo_loss(
                batch['old_log_probs'],
                new_lp,
                batch['advantages'],
                rewards,
                vals,
                entropy
            )

            # accumulate
            total_loss += loss

            # recompute KL & value‑error for logging
            with torch.no_grad():
                kl_div = (batch['old_log_probs'].exp() * (batch['old_log_probs'] - new_lp)).sum(-1).mean()
                value_error = (vals - rewards).abs().mean()

            # Log per‑step metrics
            self.log_dict({
                'train/ppo_policy_loss': policy_loss,
                'train/ppo_value_loss':  value_loss,
                'train/ppo_entropy':     entropy.mean(),
                'train/ppo_advantage':   advantages.mean(),
                'train/ppo_total_loss':  loss,
                # newly added:
                'train/ppo_kl':          kl_div,
                'train/value_error':     value_error,
            }, on_step=True, on_epoch=False, prog_bar=True)

            logger.info(
                f"[PPO] Epoch {epoch+1}: "
                f"policy_loss={policy_loss.item():.4f}, "
                f"value_loss={value_loss.item():.4f}, "
                f"entropy={entropy.mean().item():.4f}, "
                f"kl={kl_div.item():.4f}, "
                f"val_error={value_error.item():.4f}, "
                f"total_loss={loss.item():.4f}"
            )

        # 6) Finalize
        avg_loss = total_loss / self.ppo_epochs
        logger.info(f"[PPO] Average loss over {self.ppo_epochs} epochs: {avg_loss.item():.4f}")
        self.log('train/avg_ppo_loss', avg_loss, on_epoch=True, prog_bar=True)
        return avg_loss

    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW) and the learning rate scheduler (CosineAnnealingLR).
        """
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.value_head.parameters(), 'lr': CONFIG.get("value_lr", CONFIG["learning_rate"])}
        ], lr=CONFIG["learning_rate"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=CONFIG["ppo_epochs"],
            eta_min=CONFIG.get("min_lr", 1e-6)
        )
        return [self.optimizer], [scheduler]
    
    def _load_checkpoint_safely(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            # Load main model weights (excluding attention biases if needed)
            model_state_dict = {k: v for k, v in state_dict.items() 
                            if k.startswith('model.') and 
                            'relative_attention_bias' not in k}
            
            # Load value head if it exists in checkpoint
            value_head_state = {}
            for k in list(state_dict.keys()):
                if k.startswith('value_head.'):
                    value_head_state[k.replace('value_head.', '')] = state_dict.pop(k)
            
            # Load model weights
            load_res = self.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"Loaded model weights from checkpoint")
            logger.info(f"Missing keys: {load_res.missing_keys}")
            logger.info(f"Unexpected keys: {load_res.unexpected_keys}")
            
            # Initialize value head
            if value_head_state:
                self.value_head.load_state_dict(value_head_state)
                logger.info("Loaded value head weights from checkpoint")
            else:
                logger.info("Initializing value head from scratch")
                self._initialize_value_head()
                
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            logger.info("Initializing model and value head from scratch")
            self._initialize_value_head()

    def _initialize_value_head(self):
        """Initialize value head using features from pretrained model"""
        try:
            with torch.no_grad():
                # Get a sample output from the model
                sample_input = torch.randint(0, self.tokenizer.vocab_size, (1, 10)).to(self.device)
                outputs = self.model(input_ids=sample_input, decoder_input_ids=sample_input)
                hidden_state = outputs.last_hidden_state.mean(dim=1)
                
                # Initialize first layer to match pretrained features
                self.value_head[0].weight.data.normal_(mean=0.0, std=0.02)
                self.value_head[0].bias.data.zero_()
                
                # Initialize second layer to produce reasonable initial values
                self.value_head[2].weight.data.normal_(mean=0.0, std=0.01)
                self.value_head[2].bias.data.fill_(0.0)
                
                logger.info("Value head initialized with pretrained-aware scheme")
        except Exception as e:
            logger.error(f"Pretrained-aware init failed: {e}")
            # Fallback to simple init
            for layer in self.value_head:
                if hasattr(layer, 'weight'):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if hasattr(layer, 'bias'):
                    layer.bias.data.zero_()
            logger.info("Value head initialized with simple normal weights")

    def verify_initialization(self):
        """Verify model and value head are properly initialized"""
        logger.info("Verifying model initialization:")
        
        # 1. Check main model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
        
        # 2. Check value head parameters
        value_params = sum(p.numel() for p in self.value_head.parameters())
        logger.info(f"Value head params: {value_params:,}")
        
        # 3. Test forward pass
        try:
            test_input = torch.randint(0, self.tokenizer.vocab_size, (1, 10)).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(test_input, max_length=20)
                values = self.get_value(test_input, outputs)
            
            logger.info(f"Initialization test passed. Sample value: {values[0].item():.4f}")
            return True
        except Exception as e:
            logger.error(f"Initialization test failed: {e}")
            return False

    def on_save_checkpoint(self, checkpoint):
        """Save checkpoint in compatible format"""
        checkpoint['value_head_state_dict'] = dict(self.value_head.state_dict())
        
        # Convert config to dict for safe serialization
        checkpoint['model_config'] = {
            k: v for k, v in self.model.config.to_dict().items()
            if isinstance(v, (int, float, str, bool, dict, list))
        }
        
        # Mark as PPO checkpoint
        checkpoint['checkpoint_type'] = 'ppo'
          
    def _validate_batch(self, input_ids):
        """Validate input tensors before processing"""
        try:
            if (input_ids < 0).any() or (input_ids >= self.tokenizer.vocab_size).any():
                logger.warning("Invalid input_ids detected")
                
        except Exception as e:
            logger.warning(f"Validation warning: {str(e)}")
   
    def _check_for_nans(self, tensor_dict, name=""):
        """Utility to check for NaN/Inf in tensors"""
        has_nans = False
        for k, v in tensor_dict.items():
            if torch.is_tensor(v):
                if torch.isnan(v).any():
                    logger.error(f"NaN detected in {name}.{k}!")
                    has_nans = True
                if torch.isinf(v).any():
                    logger.error(f"Inf detected in {name}.{k}!")
                    has_nans = True
        return has_nans
    
    def _validate_trajectory(self, trajectory):
        """
        Validate and log each trajectory entry:
        - input_ids, generated_tokens => 1-D LongTensors
        - old_log_probs, rewards, values => 0-D FloatTensors
        """
        required = ['input_ids','generated_tokens','old_log_probs','rewards','values']
        logger.debug(f"Validating trajectory keys: {list(trajectory.keys())}")

        # 1) Presence check
        for k in required:
            if k not in trajectory:
                logger.error(f"Trajectory missing key: '{k}'")
                return False

        # 2) Sequence dims
        for key in ['input_ids','generated_tokens']:
            t = trajectory[key]
            if not (isinstance(t, torch.Tensor) and t.dim()==1):
                logger.error(f"'{key}' invalid dim: got {t.dim()} (shape={t.shape})")
                return False
            logger.debug(f"'{key}' OK: shape={t.shape}")

        # 3) Scalar dims
        for key in ['old_log_probs','rewards','values']:
            t = trajectory[key]
            if not (isinstance(t, torch.Tensor) and t.dim()==0):
                logger.error(f"'{key}' invalid dim: got {t.dim()} (shape={t.shape})")
                return False
            logger.debug(f"'{key}' OK: value={t.item():.4f}")

        logger.info("Trajectory entry validated successfully")
        return True

    def _validate_output_ids(self, output_ids):
        """Validate generated token IDs"""
        vocab_size = self.tokenizer.vocab_size
        invalid_mask = (output_ids < 0) | (output_ids >= vocab_size)

        if invalid_mask.any():
            invalid_count = invalid_mask.sum().item()
            invalid_ids = output_ids[invalid_mask].unique().tolist()

            logger.error(
                f"Model generated invalid tokens:\n"
                f"- Count: {invalid_count}/{output_ids.numel()}\n"
                f"- Invalid IDs: {invalid_ids}\n"
                f"- Sample positions: {torch.where(invalid_mask)[0].tolist()[:5]}"
            )

            logger.info(
                f"Generation failure context:\n"
                f"- Model: {self.model_name}\n"
                f"- Vocab size: {vocab_size}\n"
                f"- Output range: {output_ids.min().item()}-{output_ids.max().item()}\n"
                f"- Most frequent invalid ID: {max(set(invalid_ids), key=invalid_ids.count)}"
            )

            raise ValueError(f"Generated {invalid_count} invalid token IDs")

    def forward(self, input_ids):
        """Generate with full validation"""
        try:
            self._validate_input_ids(input_ids)

            generation_config = {
                'max_new_tokens': min(CONFIG['max_gen_length'], 512),
                'min_length': 5,  # Add minimum length
                'do_sample': True,
                'temperature': max(0.1, min(CONFIG.get("temperature", 0.7), 1.0)),
                'top_k': max(1, min(CONFIG.get("top_k", 50), self.tokenizer.vocab_size)),
                'top_p': min(max(0.01, CONFIG.get("top_p", 0.9)), 1.0),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'output_scores': True,
                'return_dict_in_generate': True,
                'length_penalty': 1.0  # Add length penalty
            }

            logger.debug(f"Generating with config: {generation_config}")
            outputs = self.model.generate(input_ids, **generation_config)
            sequences = outputs.sequences

            self._validate_output_ids(sequences)
            return sequences

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}\nInput shape: {input_ids.shape}")
            raise

    def compute_log_probs_and_entropy(self, input_ids, generated_tokens):
        """Safe logprob calculation with validation"""
        try:
            self._validate_input_ids(input_ids)
            self._validate_output_ids(generated_tokens)

            if generated_tokens.dim() != 2:
                logger.error(f"Expected 2D generated tokens, got {generated_tokens.dim()}D")
                raise ValueError("Invalid generated tokens dimension")

            # Get decoder inputs (shift right) and labels (shift left)
            decoder_input_ids = generated_tokens[:, :-1]  # Remove last token
            labels = generated_tokens[:, 1:]  # Remove first token

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                use_cache=False
            )
            logits = outputs.logits

            # Validate logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_count = torch.isnan(logits).sum().item()
                inf_count = torch.isinf(logits).sum().item()
                logger.error(
                    f"Invalid logits detected:\n"
                    f"- NaN count: {nan_count}\n"
                    f"- Inf count: {inf_count}\n"
                    f"- Logits range: [{logits.min().item()}, {logits.max().item()}]"
                )
                raise ValueError("NaN/Inf in logits")

            # Apply masking and clamping
            logits = self.apply_vocab_masking(logits)
            logits = torch.clamp(logits, min=-50, max=50)
            
            # Calculate log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.nan_to_num(log_probs, nan=-100.0, neginf=-100.0, posinf=0.0)

            # Gather log probs for actual generated tokens
            token_log_probs = log_probs.gather(
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            token_log_probs = torch.nan_to_num(token_log_probs, nan=-100.0, neginf=-100.0)

            # Mask out padding
            pad_id = self.tokenizer.pad_token_id
            nonpad = (labels != pad_id).float()
            token_log_probs = token_log_probs * nonpad

            # Average log prob per sequence
            lengths = nonpad.sum(dim=1).clamp(min=1)
            avg_log_prob = token_log_probs.sum(dim=1) / lengths
            avg_log_prob = torch.clamp(avg_log_prob, max=0.0)

            # Calculate entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1) * nonpad
            entropy = entropy.sum(dim=1) / lengths
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

            logger.debug(
                f"[LOGPROB] avg: {avg_log_prob.mean().item():.4f}, "
                f"min: {avg_log_prob.min().item():.4f}, "
                f"max: {avg_log_prob.max().item():.4f}"
            )
            logger.debug(
                f"[ENTROPY] avg: {entropy.mean().item():.4f}, "
                f"min: {entropy.min().item():.4f}, "
                f"max: {entropy.max().item():.4f}"
            )

            return avg_log_prob, entropy

        except Exception as e:
            logger.error(
                f"Logprob calculation failed:\n"
                f"Input shape: {input_ids.shape}\n"
                f"Generated shape: {generated_tokens.shape}\n"
                f"Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}\n"
                f"Error: {str(e)}"
            )
            raise      

    def safe_decode(self, token_ids):
        """Decode with validation and error recovery"""
        try:
            if not isinstance(token_ids, (list, torch.Tensor)):
                logger.error(f"Expected list/tensor, got {type(token_ids)}")
                raise TypeError("Invalid input type")

            ids_list = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
            valid_ids = []
            invalid_ids = []

            for idx, tid in enumerate(ids_list):
                if 0 <= tid < self.tokenizer.vocab_size:
                    valid_ids.append(tid)
                else:
                    invalid_ids.append((idx, tid))

            if invalid_ids:
                logger.warning(
                    f"Found {len(invalid_ids)} invalid tokens during decoding:\n"
                    f"Sample invalid positions/IDs: {invalid_ids[:5]}\n"
                    f"Replacing with pad_token_id ({self.tokenizer.pad_token_id})"
                )
                valid_ids = [tid if 0 <= tid < self.tokenizer.vocab_size
                             else self.tokenizer.pad_token_id
                             for tid in ids_list]

            return self.tokenizer.decode(valid_ids, skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Decoding failed: {str(e)}\nToken IDs: {token_ids[:20]}...")
            return "[DECODING_ERROR]"

    def training_step(self, batch, batch_idx):
        try:
            # 1) Validate input
            input_ids = batch["input_ids"]
            self._validate_input_ids(input_ids)

            orig_texts = [str(x) for x in batch["original_ending"]]
            edit_texts = [str(x) for x in batch["edited_ending"]]
            batch_size = input_ids.size(0)

            # 2) Generate output tokens
            gen_tokens = self.forward(input_ids)
            self._validate_output_ids(gen_tokens)

            logger.debug(
                f"Shapes before logprob calculation:\n"
                f"Input IDs: {input_ids.shape}\n"
                f"Generated tokens: {gen_tokens.shape}\n"
                f"Decoder input IDs: {gen_tokens[:, :-1].shape}\n"
                f"Labels: {gen_tokens[:, 1:].shape}"
            )

            # 3) Compute log-probs, entropy, and values from decoder
            old_lp, entropy = self.compute_log_probs_and_entropy(input_ids, gen_tokens)
            values = self.get_value(input_ids, gen_tokens)

            # 4) Move to CPU
            gen_tokens, old_lp, entropy, values = (
                gen_tokens.detach().cpu(),
                old_lp.detach().cpu(),
                entropy.detach().cpu(),
                values.detach().cpu()
            )

            # 5) Check for NaNs or Infs
            if self._check_for_nans({
                'gen_tokens': gen_tokens,
                'old_lp': old_lp,
                'entropy': entropy,
                'values': values
            }, name=f"batch_{batch_idx}_pre_reward"):
                logger.warning(f"[Batch {batch_idx}] NaN/Inf in model outputs, skipping")
                return None

            # 6) Compute rewards
            gen_texts = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            rewards = self.calculate_rewards(gen_texts, edit_texts, orig_texts).cpu()
            # reward statistics
            self.log('train/reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True)
            self.log('train/reward_std',  rewards.std(),  on_step=True, on_epoch=True)
            self.log('train/reward_min',  rewards.min(),  on_step=True)
            self.log('train/reward_max',  rewards.max(),  on_step=True)

            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                logger.error(f"[Batch {batch_idx}] Invalid rewards detected")
                rewards = torch.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0)

            # 7) Prepare for buffer
            old_lp = old_lp.unsqueeze(0) if old_lp.dim() == 0 else old_lp
            values = values.unsqueeze(0) if values.dim() == 0 else values

            # 8) Buffer trajectories
            valid_trajectories = 0
            for i in range(batch_size):
                traj = {
                    "input_ids": input_ids[i].cpu(),
                    "generated_tokens": gen_tokens[i],
                    "old_log_probs": old_lp[i] if i < old_lp.size(0) else torch.tensor(-1e1),
                    "rewards": rewards[i] if i < rewards.size(0) else torch.tensor(0.0),
                    "values": values[i] if i < values.size(0) else torch.tensor(0.0)
                }
                if self._validate_trajectory(traj):
                    self.trajectory_buffer.append(traj)
                    self.buffer_token_count += traj["generated_tokens"].numel()
                    valid_trajectories += 1

            logger.info(
                f"[Batch {batch_idx}] Buffered {valid_trajectories}/{batch_size} trajectories, "
                f"total tokens={self.buffer_token_count}"
            )

            self.log_dict({
                "train/reward": rewards.mean(),
                "train/entropy": entropy.mean(),
                "train/buffer_size": len(self.trajectory_buffer),
                "train/buffer_tokens": self.buffer_token_count,
            }, on_step=True, batch_size=batch_size)

            # 9) PPO update
            if self.buffer_token_count >= self.max_trajectory_length:
                logger.info(f"Buffer full ({self.buffer_token_count} tokens): running PPO update")
                try:
                    opt = self.optimizers()
                    opt.zero_grad()
                    ppo_loss = self.update_ppo()

                    if torch.isnan(ppo_loss) or torch.isinf(ppo_loss):
                        raise ValueError(f"Invalid PPO loss: {ppo_loss.item()}")

                    self.manual_backward(ppo_loss)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    # gradient health
                    self.log('train/grad_norm', grad_norm, on_step=True)
                    logger.debug(f"Clipped gradients, norm={grad_norm:.4f}")
                    opt.step()

                    return ppo_loss
                except Exception as e:
                    logger.error(f"PPO update failed: {e}\n{traceback.format_exc()}")
                    return None
                finally:
                    self.trajectory_buffer.clear()
                    self.buffer_token_count = 0
                    torch.cuda.empty_cache()
                    gc.collect()

            return None

        except Exception as e:
            logger.error(
                f"Training step failed at batch {batch_idx}:\n"
                f"Error: {str(e)}\n"
                f"Input shape: {input_ids.shape if 'input_ids' in locals() else 'N/A'}\n"
                f"Generated shape: {gen_tokens.shape if 'gen_tokens' in locals() else 'N/A'}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            self.trajectory_buffer.clear()
            self.buffer_token_count = 0
            torch.cuda.empty_cache()
            gc.collect()
            return None
