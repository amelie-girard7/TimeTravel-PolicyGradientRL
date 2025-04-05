# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/models/model_ppo.py
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

logger = logging.getLogger(__name__)

class FlanT5PPOFineTuner(pl.LightningModule):
    
    def __init__(self, model_name, model_dir, file_label=""):
        super().__init__()
        self.save_hyperparameters()

        # Store paths as attributes
        self.model_dir = Path(model_dir)
        self.file_label = file_label
        
        # Model setup
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Enhanced PPO components
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        # Initialize value head properly if not loading from checkpoint
        if not hasattr(self, 'loaded_from_checkpoint'):
            for layer in self.value_head:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                    
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
    

    def calculate_gae(self, rewards, values, done_mask=None):
        """Generalized Advantage Estimation"""
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        if done_mask is None:
            done_mask = torch.ones_like(rewards)

        for t in reversed(range(batch_size)):
            delta = rewards[t] + self.gamma * values[t+1] * done_mask[t] - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * done_mask[t] * last_advantage
        return advantages

    def calculate_ppo_loss(self, old_log_probs, new_log_probs, advantages, rewards, values):
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, rewards)
        entropy = -(torch.exp(new_log_probs) * new_log_probs).mean()
        
        # Additional metrics
        kl_div = (old_log_probs - new_log_probs).mean()
        clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
        
        self.log_dict({
            'train/policy_loss': policy_loss,
            'train/value_loss': value_loss,
            'train/entropy': entropy,
            'train/kl_divergence': kl_div,
            'train/clip_frac': clip_frac,
            'train/value_error': (values - rewards).abs().mean(),
        }, prog_bar=True)
        
        return policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

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
        elif CONFIG["ppo_experiment"] == "SCST":
            rewards = self.calculate_scst_rewards(generated_texts, edited_endings, original_endings)
        else:
            rewards = score_pred_edited

        if CONFIG.get("objective_clipping", False):
            rewards = torch.clamp(rewards, min=0.0)
        return rewards

    def calculate_scst_rewards(self, generated_texts, edited_endings, original_endings):
        # Generate greedy outputs for baseline
        input_ids = self.tokenizer(generated_texts, return_tensors="pt", padding=True).input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float().to(self.device)
        
        greedy_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_length=CONFIG['max_gen_length']
        )
        greedy_texts = self.tokenizer.batch_decode(greedy_outputs, skip_special_tokens=True)
        
        # Calculate scores
        score_sampled_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)
        score_sampled_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings)
        score_greedy_edited = self.metrics_evaluator.calculate_score(greedy_texts, edited_endings)
        score_greedy_original = self.metrics_evaluator.calculate_score(greedy_texts, original_endings)
        
        # Compute SCST reward
        delta_m1_sampled = score_sampled_edited - score_sampled_original
        delta_m1_greedy = score_greedy_edited - score_greedy_original
        return (score_sampled_edited + delta_m1_sampled) - (score_greedy_edited + delta_m1_greedy)

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

    def update_ppo(self):
        # Find max lengths for all relevant tensors
        max_input_len = max(t['input_ids'].size(1) for t in self.trajectory_buffer)
        max_gen_len = max(t['generated_tokens'].size(1) for t in self.trajectory_buffer)
        
        # Pad all sequences to their respective max lengths
        padded_trajectories = []
        for t in self.trajectory_buffer:
            # Pad input sequences
            input_pad_amount = max_input_len - t['input_ids'].size(1)
            padded_input_ids = F.pad(t['input_ids'], (0, input_pad_amount), value=self.tokenizer.pad_token_id)
            padded_attention_mask = F.pad(t['attention_mask'], (0, input_pad_amount), value=0)
            
            # Pad generated sequences
            gen_pad_amount = max_gen_len - t['generated_tokens'].size(1)
            padded_generated_tokens = F.pad(t['generated_tokens'], (0, gen_pad_amount), value=self.tokenizer.pad_token_id)
            
            padded_item = {
                'old_log_probs': t['old_log_probs'],
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask,
                'generated_tokens': padded_generated_tokens,
                'rewards': t['rewards'],
                'values': t['values'],
                'advantages': t['advantages']
            }
            padded_trajectories.append(padded_item)

        # Now safely concatenate
        batch = {
            k: torch.cat([t[k] for t in padded_trajectories])
            for k in padded_trajectories[0].keys()
        }
        

        total_loss = 0
        for _ in range(self.ppo_epochs):
            # Forward pass - use padded input
            generated_tokens, logits = self.forward(batch['input_ids'], batch['attention_mask'])
            
            # Calculate new log probs - need to handle padding here too
            logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
            logits = self.apply_vocab_masking(logits)
            
            # Handle potential padding in labels
            labels_for_indexing = generated_tokens[:, 1:].contiguous()
            token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)
            padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
            new_log_probs = (token_log_probs * padding_mask.float()).sum(dim=1)
            
            # Get new values
            new_values = self.get_value(batch['input_ids'], batch['attention_mask'])
            
            # Calculate loss
            loss = self.calculate_ppo_loss(
                batch['old_log_probs'],
                new_log_probs,
                batch['advantages'],
                batch['rewards'],
                new_values
            )
            
            total_loss += loss
            
            # Log metrics
            self.log('train/policy_loss', loss[0], on_step=True)
            self.log('train/value_loss', loss[1], on_step=True)
            self.log('train/entropy', loss[2], on_step=True)
            self.log('train/avg_reward', batch['rewards'].mean(), on_step=True)

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