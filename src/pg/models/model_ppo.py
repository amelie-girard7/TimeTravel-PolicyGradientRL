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
        Modified GAE calculation for text generation tasks:
        - Removed episodic assumption (treats each sample independently)
        - Added value normalization
        """
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Reverse calculation without episode boundaries
        for t in reversed(range(batch_size)):
            delta = rewards[t] + self.gamma * values[t+1] - values[t]
            last_advantage = delta + self.gamma * self.lam * last_advantage
            advantages[t] = last_advantage
            
        # Normalize advantages per batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def calculate_ppo_loss(self, old_log_probs, new_log_probs, advantages, rewards, values, entropy):
        """
        Computes the combined PPO loss containing:
        1. Clipped policy gradient loss
        2. Value function loss
        3. Entropy bonus
        
        Args:
            old_log_probs: Log probabilities from BEFORE policy update (detached)
            new_log_probs: Current policy's log probabilities
            advantages: Estimated advantages (GAE)
            rewards: Actual observed rewards
            values: Current value function estimates
            entropy: Policy entropy measure
        
        Returns:
            loss: Combined loss for backpropagation
            policy_loss: Clipped surrogate objective component
            value_loss: Value function MSE component
        """
        
        # --- 1. Policy Loss (Clipped Objective) ---
        # Calculate probability ratio (new policy / old policy)
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        # shape: [batch_size] or [batch_size, seq_len] for per-token
        
        # Unclipped objective
        surr1 = ratio * advantages  # Original policy gradient
        
        # Clipped objective
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon,  # Lower bound 
            1.0 + self.clip_epsilon   # Upper bound 
        ) * advantages
        
        # Take minimum of clipped vs unclipped
        policy_loss = -torch.min(surr1, surr2).mean()  
        # Negative sign because we maximize rewards (minimize negative rewards)
        
        # --- 2. Value Function Loss ---
        # MSE between predicted values and actual returns
        value_loss = F.mse_loss(values, rewards)  
        # Shapes: values=[batch_size], rewards=[batch_size]
        
        # --- 3. Entropy Bonus ---
        # Encourages exploration by penalizing low entropy
        entropy_loss = entropy.mean()  
        # Scalar value (averaged over batch)
        
        # --- 4. Combined Loss ---
        # Weighted sum of all components:
        loss = (
            policy_loss                    # Primary policy gradient term
            + self.value_coef * value_loss  # Value function accuracy 
            - self.entropy_coef * entropy_loss  # Exploration bonus 
        )
        
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
        Enhanced reward calculation with:
        - Stronger contrast between edited and original endings
        - Margin enforcement
        - Debug logging
        """
        # Always calculate both similarities
        sim_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        sim_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()
        
        # Log the raw similarities for debugging
        self.log_dict({
            'train/raw_sim_edited': sim_edited.mean(),
            'train/raw_sim_original': sim_original.mean(),
        }, prog_bar=True)
        
        if CONFIG["ppo_experiment"] == "contrastive_ratio":
            # Strong contrastive reward with margin enforcement
            margin = CONFIG.get("reward_margin", 0.2)  # Configurable margin
            contrastive_reward = sim_edited - sim_original
            
            # Apply margin condition with penalty
            rewards = torch.where(
                sim_edited > sim_original + margin,
                contrastive_reward * 2.0,  # Bonus for clear improvement
                contrastive_reward - 1.0    # Penalty for failing margin
            )
            
        elif CONFIG["ppo_experiment"] == "delta_m1":
            # Original implementation but with logging
            delta_m1 = sim_edited - sim_original
            rewards = sim_edited + delta_m1
        else:
            # Default case - just similarity to edited ending
            rewards = sim_edited
        
        # Log final rewards
        self.log('train/raw_rewards', rewards.mean(), prog_bar=True)
        
        return rewards

    def compute_log_probs_and_entropy(self, input_ids, attention_mask, generated_tokens, cached_states=None):
        """
        Computes token-level log probabilities and entropy for PPO using teacher forcing.
        Critical for calculating:
        1. Probability ratios (new vs old policy)
        2. Policy entropy (for exploration bonus)
        
        Args:
            input_ids: Source input tokens [batch_size, src_seq_len]
            attention_mask: Source attention mask [batch_size, src_seq_len]
            generated_tokens: Model's generated output [batch_size, tgt_seq_len]
            cached_states: Optional cached key-value states for efficiency
            
        Returns:
            sequence_log_prob_sum: Sum of log probs per sequence [batch_size]
            entropy: Average entropy per sequence [batch_size]
            cached_states: Updated key-value states for future reuse
        """
        
        # --- 1. Prepare Decoder Inputs/Labels ---
        # Teacher forcing: Use generated tokens as decoder input (shifted right)
        decoder_input_ids = generated_tokens[:, :-1].contiguous()  # Remove last token
        labels = generated_tokens[:, 1:].contiguous()  # Shift left for target
        

        # TODO: ENSURE THIS IS NEEDED. WE CAN USE THE GENERATE FUNCTION
        # --- 2. Forward Pass ---
        if cached_states is not None:
            # Use cached key-value states for efficient incremental decoding
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=cached_states,  # Reuse previous computations
                use_cache=True,
                output_hidden_states=True
            )
        else:
            # Full forward pass (first call or cache disabled)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,  # Still return states for potential caching
                output_hidden_states=True
            )
            cached_states = outputs.past_key_values  # Store for future steps
        
        # --- 3. Log Probability Calculation ---
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Convert to log probabilities (numerically stable)
        log_probs_all = torch.log_softmax(logits, dim=-1)
        
        # Gather log probs only for actual generated tokens
        token_log_probs = log_probs_all.gather(
            dim=-1, 
            index=labels.unsqueeze(-1)  # Add dimension for gathering
        ).squeeze(-1)  # [batch_size, seq_len]
        
        # Sum log probs across sequence for each sample
        sequence_log_prob_sum = token_log_probs.sum(dim=1)  # [batch_size]
        
        # --- 4. Entropy Calculation ---
        probs_all = torch.softmax(logits, dim=-1)
        entropy_per_token = -(probs_all * log_probs_all).sum(dim=-1)  # [batch_size, seq_len]
        entropy = entropy_per_token.mean(dim=1)  # Average over sequence [batch_size]
        
        return sequence_log_prob_sum, entropy, cached_states   

    def update_ppo(self):
        """
        Enhanced PPO update with:
        - Better advantage calculation
        - More detailed logging
        - Gradient clipping
        """
        batch = self.prepare_ppo_batch(self.trajectory_buffer)
        rewards = batch['rewards']
        values = batch['values'].squeeze(-1)
        
        # Add bootstrap value
        values = torch.cat([values, torch.zeros(1, device=values.device)])
        
        # Simplified done mask (all zeros for text generation)
        done_mask = torch.zeros_like(rewards)
        
        # Calculate advantages
        advantages = self.calculate_gae(rewards, values, done_mask)
        batch['advantages'] = advantages

        total_loss = 0
        for epoch in range(self.ppo_epochs):
            # Forward pass
            new_log_probs, entropy, _ = self.compute_log_probs_and_entropy(
                batch['input_ids'], batch['attention_mask'], batch['generated_tokens']
            )

            # Calculate losses
            loss, policy_loss, value_loss = self.calculate_ppo_loss(
                batch['old_log_probs'],
                new_log_probs,
                batch['advantages'],
                batch['rewards'],
                self.get_value(batch['input_ids'], batch['attention_mask']).squeeze(-1),
                entropy
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                CONFIG.get("gradient_clip_val", 0.5)
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_head.parameters(),
                CONFIG.get("gradient_clip_val", 0.5)
            )
            
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            self.log_dict({
                'ppo/epoch_policy_loss': policy_loss.item(),
                'ppo/epoch_value_loss': value_loss.item(),
                'ppo/epoch_entropy': entropy.mean().item(),
                'ppo/epoch_advantage': batch['advantages'].mean().item(),
            })

        return total_loss / self.ppo_epochs

    def training_step(self, batch, batch_idx):
        """
        Enhanced training step with:
        - Removed reward normalization
        - Added sample logging
        - Better error handling
        """
        try:
            # --- 1. Prepare Input Data ---
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            original_endings = [str(oe) for oe in batch['original_ending']]

            # --- 2. Policy Rollout ---
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],
                do_sample=True,
                temperature=CONFIG.get("temperature", 0.7),
                top_k=CONFIG.get("top_k", 50),  # Added for better sampling
                top_p=CONFIG.get("top_p", 0.9),  # Added for better sampling
                output_scores=True,
                return_dict_in_generate=True
            )
            
            generated_tokens = outputs.sequences
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Log some samples periodically
            if batch_idx % 100 == 0:
                for i in range(min(2, len(generated_texts))):  # Log first 2 samples
                    logger.info(f"\nSample {i+1}:")
                    logger.info(f"Original: {original_endings[i]}")
                    logger.info(f"Edited: {edited_endings[i]}")
                    logger.info(f"Generated: {generated_texts[i]}\n")

            # --- 3. Reward Computation ---
            rewards = self.calculate_rewards(generated_texts, edited_endings, original_endings)
            
            # Removed reward normalization - using raw rewards now
            rewards = rewards.to(self.device)

            # --- 4. Value Estimation ---
            values = self.get_value(input_ids, attention_mask)

            # --- 5. Policy Probabilities ---
            old_log_probs, entropy, _ = self.compute_log_probs_and_entropy(
                input_ids, attention_mask, generated_tokens
            )

            # --- 6. Store Trajectory ---
            trajectory_item = {
                'old_log_probs': old_log_probs.detach(),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'generated_tokens': generated_tokens,
                'rewards': rewards,
                'values': values.detach()
            }
            
            self.trajectory_buffer.append(trajectory_item)

            # --- 7. PPO Update ---

            # do we need the condition here 
            # if len(self.trajectory_buffer) >= self.max_trajectory_length: 
            avg_loss = self.update_ppo()
            self.trajectory_buffer = []
            self.log('train/avg_ppo_loss', avg_loss, prog_bar=True)

            return None

        except Exception as e:
            logger.error(f"Training error (batch {batch_idx}): {str(e)}")
            self.trajectory_buffer = []
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
