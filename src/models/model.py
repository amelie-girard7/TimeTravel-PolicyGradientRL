# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/models/model.py
import csv
import logging
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.metrics import MetricsEvaluator
import pandas as pd
import wandb

# Initialize a logger for debugging and output control
logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model using policy gradient reinforcement learning.
    Supports both Maximum Likelihood Estimation (MLE) and Policy Gradient (PG) training modes.
    """
    def __init__(self, model_name, model_dir, file_label=""):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()
        self.save_hyperparameters()  # Saves model_name and model_dir as hyperparameters for reference

        # Convert model_dir to a Path object for consistent file handling
        model_dir = Path(model_dir)

        # Load T5 model and tokenizer with configurations specified in CONFIG
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Set unique file paths using `file_label` to prevent overwriting
        self.val_csv_file_path = model_dir / f"validation_details{file_label}.csv"
        self.test_csv_file_path = model_dir / f"test_details{file_label}.csv"
        self.epoch_validation_details = []  # Storage for each validation epoch

        # Initialize MetricsEvaluator to handle custom scoring for rewards
        self.metrics_evaluator = MetricsEvaluator()

        # This attribute will be set in main.py to toggle between MLE and PG modes
        self.use_policy_gradient = False

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the T5 model.
        If labels are provided, calculates the loss; otherwise, returns generated tokens and logits.
        """
        if labels is not None:
            # MLE training mode with labels for loss calculation
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=False
            )
            return outputs
        else:
            # PG mode generates tokens without labels
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],
                num_beams=1,  # Greedy decoding
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

    def custom_loss(self, outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to the calculation.
        """
        logits_flat = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten targets to [batch_size * seq_length]
        differential_weights_flat = differential_weights.view(-1)  # Flatten weights to match sequence length [batch_size * seq_length]  

        # Compute the standard loss function without reduction to get a loss value per token.
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Apply the differential weights to each token's loss.
        weighted_loss_per_token = loss_per_token * differential_weights_flat

        # Calculate the mean of the weighted losses to get a single scalar representing the batch's loss.
        mean_weighted_loss = weighted_loss_per_token.mean()
        
        return mean_weighted_loss

    def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards):
        """
        Calculates policy gradient loss based on generated tokens and rewards.
        """
        # Stack logits along the sequence dimension and apply log softmax
        logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
        logits = self.apply_vocab_masking(logits)  # Apply masking to stacked logits

        # Gather the log probabilities for the generated tokens
        labels_for_indexing = generated_tokens[:, 1:].contiguous()
        token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

        # Create a mask to ignore padding tokens
        padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
        token_log_probs = token_log_probs * padding_mask.float()

        # Sum log probabilities across the sequence dimension
        sequence_log_prob_sum = token_log_probs.sum(dim=1)

        # Calculate policy gradient loss
        return -(rewards * sequence_log_prob_sum).mean()

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        if self.use_policy_gradient:
            # Policy Gradient (PG) training mode
            generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]

            # Calculate rewards for generated texts
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
            rewards = scores - CONFIG["baseline_score"]

            # Calculate PG loss
            pg_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards)
            
            # Log Policy Gradient-specific training metrics
            self.log('training_pg_loss', pg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('training_pg_reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return pg_loss  # Return PG loss for optimization

        else:
            # Maximum Likelihood Estimation (MLE) training mode
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            masked_logits = self.apply_vocab_masking(outputs.logits)  # Apply masking to logits

            # Check the 'use_custom_loss' config and whether 'differential_weights' is in the batch
            if CONFIG['use_custom_loss'] and 'differential_weights' in batch:
                mle_train_loss = self.custom_loss(masked_logits, batch['labels'], batch['differential_weights'])
            else:
                mle_train_loss = outputs.loss
            
            # Log MLE training loss
            self.log('training_mle_loss', mle_train_loss, on_epoch=True, prog_bar=True, logger=True)

            # Calculate and log average score for generated texts
            generated_texts = self.tokenizer.batch_decode(masked_logits.argmax(-1), skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
            score_mean = scores.mean()
            self.log('training_mle_score_mean', score_mean, on_epoch=True, prog_bar=True, logger=True)
            
            return mle_train_loss  # Return MLE loss for optimization

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        if self.use_policy_gradient:
            # Policy Gradient (PG) validation mode
            generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]

            # Check for non-empty generated texts and references
            non_empty_indices = [i for i, text in enumerate(generated_texts) if text.strip()]
            if not non_empty_indices:
                logger.warning("All generated texts are empty in this batch; skipping ROUGE calculation.")
                return torch.tensor(0.0, device=self.device)  # Return a placeholder loss for empty cases

            # Filter lists to only include non-empty elements
            generated_texts = [generated_texts[i] for i in non_empty_indices]
            edited_endings = [edited_endings[i] for i in non_empty_indices]

            # Calculate rewards and PG validation loss
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
            rewards = scores - CONFIG["baseline_score"]
            pg_val_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards)

            # Log PG validation loss and reward mean
            self.log('validation_pg_loss', pg_val_loss, on_epoch=True, prog_bar=True, logger=True)
            self.log('validation_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)

            # Ensure logging of `validation_mle_loss` even in PG mode, for checkpointing compatibility
            self.log('validation_mle_loss', pg_val_loss, on_epoch=True, prog_bar=False, logger=True)  # Log for compatibility

            # Save PG validation details for end-of-epoch logging
            for i in range(len(generated_texts)):
                self.epoch_validation_details.append({
                    'batch_idx': batch_idx,
                    'Epoch': self.current_epoch,
                    'Premise': batch['premise'][i],
                    'Initial': batch['initial'][i],
                    'Counterfactual': batch['counterfactual'][i],
                    'Original Ending': batch['original_ending'][i],
                    'Edited Ending': edited_endings[i],
                    'Generated Text': generated_texts[i],
                    'Reward': rewards[i].item(),
                    'pg_val_loss': pg_val_loss.item(),
                    'Mode': "Policy Gradient"
                })
            return pg_val_loss  # Return PG validation loss for logging

        else:
            # MLE validation mode
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            mle_val_loss = outputs.loss  # Calculate MLE loss directly from outputs

            # Log MLE validation loss
            self.log('validation_mle_loss', mle_val_loss, on_epoch=True, prog_bar=True, logger=True)

            # Decode and calculate scores for MLE validation using `generate` (for proper inference behavior)
            generated_texts = self.tokenizer.batch_decode(
                self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=CONFIG['max_gen_length']),
                skip_special_tokens=True
            )
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            
            # Filter empty generated texts for ROUGE calculation
            non_empty_indices = [i for i, text in enumerate(generated_texts) if text.strip()]
            if not non_empty_indices:
                logger.warning("All generated texts are empty in this batch; skipping ROUGE calculation.")
                return mle_val_loss
            
            # Apply filtering
            generated_texts = [generated_texts[i] for i in non_empty_indices]
            edited_endings = [edited_endings[i] for i in non_empty_indices]

            # Calculate scores for generated texts
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
            score_mean = scores.mean()
            self.log('validation_mle_score_mean', score_mean, on_epoch=True, prog_bar=True, logger=True)

            # Save MLE validation details for end-of-epoch logging
            for i in range(len(generated_texts)):
                self.epoch_validation_details.append({
                    "batch_idx": batch_idx,
                    "Epoch": self.current_epoch,
                    "Premise": batch['premise'][i],
                    "Initial": batch['initial'][i],
                    "Counterfactual": batch['counterfactual'][i],
                    "Original Ending": batch['original_ending'][i],
                    "Edited Ending": edited_endings[i],
                    "Generated Text": generated_texts[i],
                    "Score": scores[i].item(),
                    "mle_val_loss": mle_val_loss.item(),
                    "Mode": "MLE"
                })
            return mle_val_loss  # Return MLE validation loss for logging

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations at the end of the validation epoch, saving to CSV.
        """
        csv_file_path = self.test_csv_file_path if test_flag else self.val_csv_file_path
        if self.epoch_validation_details:
            self.log_to_csv(csv_file_path, self.epoch_validation_details)
        else:
            logger.info("No validation details available for logging.")
        self.cleanup_epoch_data()

    def log_to_csv(self, csv_file_path, details):
        """
        Logs the validation or test results into a CSV file.
        """
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(details)

    def determine_csv_path(self, test_flag):
        return self.test_csv_file_path if test_flag else self.val_csv_file_path

    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set,
        calculate the loss, and optionally generate text.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(test_flag=True)

    def cleanup_epoch_data(self):
        """
        Cleans up data collected during the epoch to prepare for the next epoch.
        """
        self.epoch_validation_details.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
