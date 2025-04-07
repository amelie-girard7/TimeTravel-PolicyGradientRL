# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/models/model_ppo.py
import csv
import logging
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.pg.utils.configppo import CONFIG
from src.pg.utils.metrics import MetricsEvaluator
import pandas as pd
import wandb

logger = logging.getLogger(__name__)

class FlanT5PPOFineTuner(pl.LightningModule):
    
    def __init__(self, model_name, model_dir, file_label=""):
        super().__init__()
        self.save_hyperparameters()

        self.model_dir = Path(model_dir)
        self.file_label = file_label

        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Initialize value head explicitly
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

        # Explicitly load pretrained value head (MLE checkpoint)
        if CONFIG.get("ppo_from_checkpoint", False) and CONFIG["ppo_checkpoint_path"]:
            supervised_checkpoint = torch.load(CONFIG["ppo_checkpoint_path"], map_location=self.device)
            state_dict = supervised_checkpoint.get('state_dict', supervised_checkpoint)
            value_head_weights = {k.replace('value_head.', ''): v for k, v in state_dict.items() if 'value_head.' in k}
            if value_head_weights:
                self.value_head.load_state_dict(value_head_weights, strict=False)
                logger.info("Loaded pretrained value head from supervised (MLE) checkpoint.")
            else:
                logger.warning("Value head weights not found in checkpoint; training from scratch.")
        else:
            logger.info("Training value head from scratch.")

        self.trajectory_buffer = []

        # Hyperparameters
        self.clip_epsilon = CONFIG["ppo_clip_epsilon"]
        self.ppo_epochs = CONFIG["ppo_epochs"]
        self.entropy_coef = CONFIG["entropy_coef"]
        self.max_trajectory_length = CONFIG["max_trajectory_length"]
        self.gamma = CONFIG["gamma"]
        self.lam = CONFIG["lambda"]
        self.value_coef = CONFIG.get("value_coef", 0.5)

        # Metrics and paths
        self.metrics_evaluator = MetricsEvaluator()
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"
        self.epoch_validation_details = []
        self.epoch_test_details = []


    def forward(self, input_ids, attention_mask):
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
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.value_head(encoder_outputs.last_hidden_state.mean(dim=1))
    

    def calculate_gae(self, rewards, values, done_mask):
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(batch_size)):
            if done_mask[t]:
                last_advantage = 0
            delta = rewards[t] + self.gamma * values[t+1] * (1 - done_mask[t]) - values[t]
            last_advantage = delta + self.gamma * self.lam * (1 - done_mask[t]) * last_advantage
            advantages[t] = last_advantage

        return advantages


    def calculate_ppo_loss(self, old_log_probs, new_log_probs, advantages, rewards, values, entropy):
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, rewards)

        entropy_loss = entropy.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        return loss, policy_loss, value_loss


    def apply_vocab_masking(self, logits):
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
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        
        if CONFIG["ppo_experiment"] == "delta_m1":
            score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
        # elif CONFIG["ppo_experiment"] == "SCST":
        #     rewards = self.calculate_scst_rewards(generated_texts, edited_endings, original_endings)
        else:
            rewards = score_pred_edited

        # if CONFIG.get("objective_clipping", False):
        #     rewards = torch.clamp(rewards, min=0.0)
        return rewards

    def update_ppo(self):
        # Ensure trajectories are collected beforehand
        batch = self.prepare_ppo_batch(self.trajectory_buffer)

        total_loss = 0
        for epoch in range(self.ppo_epochs):
            # Forward pass (without regenerating new trajectories)
            new_log_probs, entropy = self.compute_log_probs_and_entropy(
                batch['input_ids'], batch['attention_mask'], batch['generated_tokens']
            )

            # Get new values
            new_values = self.get_value(batch['input_ids'], batch['attention_mask']).squeeze(-1)

            # Advantage normalization (important!)
            advantages = batch['advantages']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO loss calculation
            loss, policy_loss, value_loss = self.calculate_ppo_loss(
                batch['old_log_probs'],
                new_log_probs,
                advantages,
                batch['rewards'],
                new_values,
                entropy
            )

            # Optimize explicitly
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), CONFIG.get("gradient_clip_val", 0.5))
            self.optimizer.step()

            total_loss += loss.item()

            # Enhanced logging per PPO epoch
            self.log_dict({
                'ppo/total_loss': loss.item(),
                'ppo/policy_loss': policy_loss.item(),
                'ppo/value_loss': value_loss.item(),
                'ppo/entropy': entropy.mean().item(),
                'ppo/avg_advantage': advantages.mean().item(),
                'ppo/avg_value_pred': new_values.mean().item(),
                'ppo/avg_reward': batch['rewards'].mean().item(),
            }, prog_bar=True)

        avg_total_loss = total_loss / self.ppo_epochs
        return avg_total_loss

    def training_step(self, batch, batch_idx):
        try:
            # Extract and prepare inputs
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            original_endings = [str(oe) for oe in batch['original_ending']]

            # Generate text with length control through config
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],  # Controlled by config
                do_sample=True,
                temperature=CONFIG.get("temperature", 0.7),
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_tokens = outputs.sequences
            logits = outputs.scores
            
            # Decode texts and verify lengths
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if len(generated_texts) != len(edited_endings):
                raise ValueError(f"Generated {len(generated_texts)} texts but expected {len(edited_endings)}")

            # Calculate rewards with stability checks
            rewards = self.calculate_rewards(generated_texts, edited_endings, original_endings)
            rewards = torch.clamp(rewards, -10, 10)  # Prevent extreme values
            rewards = rewards.to(self.device)

            # Get values and advantages
            values = self.get_value(input_ids, attention_mask)
            advantages = rewards - values.detach()

            # Calculate log probabilities with proper masking
            stacked_logits = torch.stack(logits, dim=1)
            log_probs = torch.log_softmax(stacked_logits, dim=-1)
            log_probs = self.apply_vocab_masking(log_probs)
            
            # Handle sequence lengths carefully
            labels = generated_tokens[:, 1:].contiguous()
            token_log_probs = log_probs.gather(
                dim=-1,
                index=labels.unsqueeze(-1)
            ).squeeze(-1)
            
            padding_mask = (labels != self.tokenizer.pad_token_id).float()
            sequence_log_prob_sum = (token_log_probs * padding_mask).sum(dim=1)

            # Store trajectory with device consistency
            trajectory_item = {
                'old_log_probs': sequence_log_prob_sum.detach(),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'generated_tokens': generated_tokens,
                'rewards': rewards,
                'values': values.detach(),
                'advantages': advantages.detach()
            }
            
            # Move all to current device
            trajectory_item = {k: v.to(self.device) for k, v in trajectory_item.items()}
            self.trajectory_buffer.append(trajectory_item)

            # Update if buffer is full
            if len(self.trajectory_buffer) >= self.max_trajectory_length:
                loss = self.update_ppo()
                self.trajectory_buffer = []  # Reset buffer
                return loss
                
            return None

        except Exception as e:
            logger.error(f"Error in training_step (batch {batch_idx}): {str(e)}", exc_info=True)
            self.trajectory_buffer = []  # Reset buffer on error
            return None  # Skip this batch

    

        return total_loss / self.ppo_epochs

    def validation_step(self, batch, batch_idx):
        # Keep your existing validation logic
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        generated_tokens, _ = self.forward(input_ids, attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Store validation details
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': batch['edited_ending'][i],
                'Generated Text': generated_texts[i]
            })
        
        # Calculate validation metrics if needed
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        rewards = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)
        self.log('val/avg_reward', rewards.mean(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Similar to validation step
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