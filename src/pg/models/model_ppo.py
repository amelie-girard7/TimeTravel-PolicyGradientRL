import logging
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.pg.utils.configppo import CONFIG
from src.pg.utils.metrics import MetricsEvaluator

logger = logging.getLogger(__name__)

class FlanT5PPOFineTuner(pl.LightningModule):
    def __init__(self, model_name, model_dir, file_label=""):
        """
        Initialize the PPO fine-tuner with T5 model, tokenizer, and value head.
        Sets up buffers, hyperparameters, and paths.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # Load T5 model configuration and model.
        config = T5Config.from_pretrained(model_name, output_attentions=CONFIG["output_attentions"])
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Build a value head for estimating state values.
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

        # Optionally load pre-trained weights for the value head.
        if CONFIG.get("ppo_from_checkpoint", False) and CONFIG["ppo_checkpoint_path"]:
            supervised_checkpoint = torch.load(CONFIG["ppo_checkpoint_path"], map_location=self.device)
            state_dict = supervised_checkpoint.get('state_dict', supervised_checkpoint)
            # Remove the 'value_head.' prefix if present.
            value_head_weights = {k.replace('value_head.', ''): v for k, v in state_dict.items() if 'value_head.' in k}
            if value_head_weights:
                self.value_head.load_state_dict(value_head_weights, strict=False)
                logger.info("Loaded pretrained value head from supervised (MLE) checkpoint.")
            else:
                logger.warning("Value head weights not found in checkpoint; training from scratch.")
        else:
            logger.info("Training value head from scratch.")

        # Buffer to collect trajectories for PPO updates.
        self.trajectory_buffer = []

        # PPO hyperparameters.
        self.clip_epsilon = CONFIG["ppo_clip_epsilon"]
        self.ppo_epochs = CONFIG["ppo_epochs"]
        self.entropy_coef = CONFIG["entropy_coef"]
        self.max_trajectory_length = CONFIG["max_trajectory_length"]
        self.gamma = CONFIG["gamma"]
        self.lam = CONFIG["lambda"]
        self.value_coef = CONFIG.get("value_coef", 0.5)

        # Setup metric evaluator and file paths.
        self.metrics_evaluator = MetricsEvaluator()
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"
        self.epoch_validation_details = []
        self.epoch_test_details = []

    def prepare_ppo_batch(self, trajectory_buffer):
        """
        Combines the trajectory buffer items into a single batch.
        Each trajectory item is concatenated along a new dimension.
        """
        batch = {
            'input_ids': torch.cat([item['input_ids'].unsqueeze(0) for item in trajectory_buffer], dim=0),
            'attention_mask': torch.cat([item['attention_mask'].unsqueeze(0) for item in trajectory_buffer], dim=0),
            'generated_tokens': torch.cat([item['generated_tokens'].unsqueeze(0) for item in trajectory_buffer], dim=0),
            'old_log_probs': torch.cat([item['old_log_probs'].unsqueeze(0) for item in trajectory_buffer], dim=0),
            'rewards': torch.cat([item['rewards'].unsqueeze(0) for item in trajectory_buffer], dim=0),
            'values': torch.cat([item['values'].unsqueeze(0) for item in trajectory_buffer], dim=0),
        }
        return batch

    def forward(self, input_ids, attention_mask):
        """
        Uses model.generate for inference.
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG['max_gen_length'],
            do_sample=True,
            temperature=CONFIG.get("temperature", 0.7),
            output_scores=True,
            return_dict_in_generate=True
        )
        return outputs.sequences, outputs.scores

    def get_value(self, input_ids, attention_mask):
        """
        Estimates the state value using the T5 encoder and the value head.
        """
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.value_head(encoder_outputs.last_hidden_state.mean(dim=1))

    def calculate_gae(self, rewards, values, done_mask):
        """
        Computes Generalized Advantage Estimation (GAE) using backward recursion.
        """
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(batch_size)):
            if done_mask[t]:
                last_advantage = 0  # Reset at episode boundaries.
            delta = rewards[t] + self.gamma * values[t+1] * (1 - done_mask[t]) - values[t]
            last_advantage = delta + self.gamma * self.lam * (1 - done_mask[t]) * last_advantage
            advantages[t] = last_advantage
        return advantages

    def calculate_ppo_loss(self, old_log_probs, new_log_probs, advantages, rewards, values, entropy):
        """
        Computes the combined PPO loss: policy loss (with clipping), value loss, and entropy loss.
        """
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, rewards)
        entropy_loss = entropy.mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        return loss, policy_loss, value_loss

    def apply_vocab_masking(self, logits):
        """
        Masks logits beyond the vocabulary size.
        """
        vocab_size = self.tokenizer.vocab_size
        if logits.dim() == 2:
            masked_logits = logits.clone()
            masked_logits[:, vocab_size:] = -float('inf')
        elif logits.dim() == 3:
            masked_logits = logits.clone()
            masked_logits[:, :, vocab_size:] = -float('inf')
        else:
            raise ValueError(f"Unexpected logits dimension: {logits.dim()}")
        return masked_logits

    def calculate_rewards(self, generated_texts, edited_endings, original_endings):
        """
        Computes rewards using the configured metric.
        For 'delta_m1', it adjusts the rewards based on the difference.
        """
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        if CONFIG["ppo_experiment"] == "delta_m1":
            score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
        else:
            rewards = score_pred_edited
        return rewards

    def compute_log_probs_and_entropy(self, input_ids, attention_mask, generated_tokens, cached_states=None):
        """
        Recomputes token-level log probabilities and entropy via teacher forcing.
        Optionally uses cached decoder states to speed up computation.
        """
        # Prepare inputs: remove the last token for decoder input and shift labels.
        decoder_input_ids = generated_tokens[:, :-1].contiguous()
        labels = generated_tokens[:, 1:].contiguous()

        if cached_states is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=cached_states,
                use_cache=True,
                output_hidden_states=True
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
                output_hidden_states=True
            )
            cached_states = outputs.past_key_values

        logits = outputs.logits
        log_probs_all = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        probs_all = torch.softmax(logits, dim=-1)
        entropy_per_token = -(probs_all * log_probs_all).sum(dim=-1)
        sequence_log_prob_sum = token_log_probs.sum(dim=1)
        entropy = entropy_per_token.mean(dim=1)
        return sequence_log_prob_sum, entropy, cached_states

    def update_ppo(self):
        """
        Performs the PPO update:
          1. Prepares the batch from the trajectory buffer.
          2. Computes rewards, values, and advantages.
          3. Runs multiple PPO epochs updating the model parameters.
        """
        # Combine trajectories into a batch.
        batch = self.prepare_ppo_batch(self.trajectory_buffer)

        # Extract rewards and values.
        rewards = batch['rewards']
        values = batch['values'].squeeze(-1)
        values = torch.cat([values, torch.zeros(1, device=values.device)])  # Bootstrap value.
        done_mask = torch.zeros_like(rewards)
        done_mask[-1] = 1.0

        advantages = self.calculate_gae(rewards, values, done_mask)
        batch['advantages'] = advantages

        total_loss = 0
        use_cache = CONFIG.get("cache_teacher_states", False)
        cached_states = None

        for epoch in range(self.ppo_epochs):
            if use_cache and cached_states is not None:
                new_log_probs, entropy, _ = self.compute_log_probs_and_entropy(
                    batch['input_ids'], batch['attention_mask'], batch['generated_tokens'],
                    cached_states=cached_states
                )
            else:
                new_log_probs, entropy, new_cached_states = self.compute_log_probs_and_entropy(
                    batch['input_ids'], batch['attention_mask'], batch['generated_tokens'], cached_states=None
                )
                if use_cache:
                    cached_states = new_cached_states
            advantages_norm = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)
            loss, policy_loss, value_loss = self.calculate_ppo_loss(
                batch['old_log_probs'],
                new_log_probs,
                advantages_norm,
                batch['rewards'],
                self.get_value(batch['input_ids'], batch['attention_mask']).squeeze(-1),
                entropy
            )
            self.optimizer.zero_grad()  # Manual optimization: zero gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), CONFIG.get("gradient_clip_val", 0.5))
            self.optimizer.step()

            total_loss += loss.item()
            # Log key PPO metrics.
            self.log_dict({
                'ppo/total_loss': loss.item(),
                'ppo/policy_loss': policy_loss.item(),
                'ppo/value_loss': value_loss.item(),
                'ppo/entropy': entropy.mean().item(),
                'ppo/avg_advantage': advantages_norm.mean().item(),
                'ppo/avg_value_pred': self.get_value(batch['input_ids'], batch['attention_mask']).squeeze(-1).mean().item(),
                'ppo/avg_reward': batch['rewards'].mean().item(),
            }, prog_bar=True)

        avg_total_loss = total_loss / self.ppo_epochs
        return avg_total_loss

    def training_step(self, batch, batch_idx):
        """
        Executes one training step:
          - Generates tokens.
          - Computes rewards and log probabilities.
          - Appends trajectory to buffer and triggers PPO update when full.
          Always returns None (manual optimization).
        """
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            original_endings = [str(oe) for oe in batch['original_ending']]

            # Generate outputs with sampling.
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],
                do_sample=True,
                temperature=CONFIG.get("temperature", 0.7),
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_tokens = outputs.sequences
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if len(generated_texts) != len(edited_endings):
                raise ValueError(f"Generated {len(generated_texts)} texts but expected {len(edited_endings)}")

            # Compute rewards and clamp extreme values.
            rewards = self.calculate_rewards(generated_texts, edited_endings, original_endings)
            rewards = torch.clamp(rewards, -10, 10).to(self.device)
            values = self.get_value(input_ids, attention_mask)
            old_log_probs, _, _ = self.compute_log_probs_and_entropy(input_ids, attention_mask, generated_tokens, cached_states=None)

            # Package trajectory data.
            trajectory_item = {
                'old_log_probs': old_log_probs.detach(),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'generated_tokens': generated_tokens,
                'rewards': rewards,
                'values': values.detach()
            }
            # Ensure data is on the proper device.
            trajectory_item = {k: v.to(self.device) for k, v in trajectory_item.items()}
            self.trajectory_buffer.append(trajectory_item)

            # When the buffer is full, trigger a PPO update.
            if len(self.trajectory_buffer) >= self.max_trajectory_length:
                _ = self.update_ppo()
                self.trajectory_buffer = []  # Clear buffer after update.

            # ALWAYS return None to signal manual optimization.
            return None

        except Exception as e:
            logger.error(f"Error in training_step (batch {batch_idx}): {str(e)}", exc_info=True)
            self.trajectory_buffer = []  # Clear buffer on error.
            return None

    def validation_step(self, batch, batch_idx):
        """
        Validation step:
          - Generates tokens.
          - Decodes and stores detailed results.
          - Logs average reward.
        """
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        generated_tokens, _ = self.forward(input_ids, attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': batch['edited_ending'][i],
                'Generated Text': generated_texts[i]
            })
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        rewards = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)
        self.log('val/avg_reward', rewards.mean(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        Test step:
          - Generates tokens.
          - Decodes outputs and saves details.
          - Logs average reward.
        """
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        generated_tokens, _ = self.forward(input_ids, attention_mask)
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

    def configure_optimizers(self):
        """
        Configures the optimizer (AdamW) and the learning rate scheduler (CosineAnnealingLR).
        """
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.value_head.parameters(), 'lr': CONFIG.get("value_lr", CONFIG["learning_rate"])}
        ], lr=CONFIG["learning_rate"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["ppo_epochs"],
            eta_min=CONFIG.get("min_lr", 1e-6)
        )
        return [optimizer], [scheduler]
