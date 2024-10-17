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

# Initialize a logger
logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model using policy gradient reinforcement learning.
    This class handles token generation, reward calculation, and policy gradient-based optimization.
    """

    def __init__(self, model_name, model_dir):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()

        # Ensure model_dir is a Path object
        model_dir = Path(model_dir)

        # Load the T5 model with the configuration, ensuring no attention is returned
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]  # Ensure attentions are not returned unless required
        )

        # Initialize the T5 model and tokenizer with the specified configuration
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # File paths for saving validation and test details
        self.val_csv_file_path = model_dir / "validation_details.csv"
        self.test_csv_file_path = model_dir / "test_details.csv"

        # Store validation step outputs for aggregation
        self.current_val_step_outputs = []
        self.epoch_validation_details = []

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass using the T5 model.
        If labels are provided, it calculates the loss (MLE loss); 
        otherwise, it returns generated tokens and logits.
        """
        print(f"\n[Forward Step] Input IDs shape: {input_ids.shape}")
        print(f"[Forward Step] Attention Mask shape: {attention_mask.shape}")
        
        if labels is not None:
            # If labels are provided, the model will calculate the MLE loss internally.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,  # T5 model computes MLE loss if labels are passed
                output_attentions=False
            )
            print(f"[Forward Step] Model is calculating loss internally.")
            return outputs
        else:
            # If no labels, generate tokens using the model and return logits
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],  # Max length for generated text
                num_beams=1,  # Greedy decoding
                output_scores=True,  # Return logits (scores) from the model
                return_dict_in_generate=True  # Return full output (sequences + logits)
            )
            
            generated_tokens = outputs.sequences
            logits = outputs.scores  # Logits for each token generated
            
            print(f"[Forward Step] Generated tokens shape: {generated_tokens.shape}")
            print(f"[Forward Step] Logits (scores) length: {len(logits)}")
            print(f"[Forward Step] Logits[0] shape (one step logits): {logits[0].shape}")
            
            return generated_tokens, logits

    def training_step(self, batch, batch_idx):
        """
        Training step to use either MLE Loss or Policy Gradient Loss based on the config.
        """
        print(f"\n[Training Step] Batch Index: {batch_idx}")
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        print(f"[Training Step] Input IDs shape: {input_ids.shape}")
        print(f"[Training Step] Attention Mask shape: {attention_mask.shape}")

        if CONFIG["use_policy_gradient"]:
            # ---- POLICY GRADIENT LOSS ----
            generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            
            # Decode the generated sequences into text
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            
            # Calculate rewards based on generated texts
            metrics_evaluator = MetricsEvaluator()
            rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
            rewards = rewards - CONFIG["baseline_score"]  # Adjust by baseline score
                       
            # Compute log probabilities for the generated tokens
            labels_for_indexing = generated_tokens[:, 1:].contiguous()  # Exclude the first token (start token)

            logits = torch.stack(logits, dim=1)  # Stack logits along the sequence dimension
            logits = torch.log_softmax(logits,dim=-1)

            token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)
            # Create a mask to ignore padding tokens
            padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
            token_log_probs = token_log_probs * padding_mask.float()
            
            # Sum log probabilities across the sequence
            sequence_log_prob_sum = token_log_probs.sum(dim=1)
            
            # Compute policy gradient loss
            loss = -(rewards * sequence_log_prob_sum)
            loss = loss.mean()  # Optimizer expects scalar loss

        else:
            # ---- MLE LOSS ----
            # Let the T5 model handle the MLE loss internally
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # T5 internally computes the MLE (cross-entropy) loss

            print(f"[Training Step] MLE Loss: {loss}")

        # Log the loss (and reward if policy gradient is used)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if CONFIG["use_policy_gradient"]:
            self.log('train_reward', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step to use either MLE Loss or Policy Gradient Loss based on the config.
        """
        print(f"\n[Validation Step] Batch Index: {batch_idx}")

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        print(f"[Validation Step] Input IDs shape: {input_ids.shape}")
        print(f"[Validation Step] Attention Mask shape: {attention_mask.shape}")

        if CONFIG["use_policy_gradient"]:
            # ---- POLICY GRADIENT LOSS ----
            generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            
            # Decode the generated sequences into text
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            
            # Calculate rewards based on generated texts
            metrics_evaluator = MetricsEvaluator()
            rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
            rewards = rewards - CONFIG["baseline_score"]
            rewards = rewards.detach()  # Detach rewards for validation (no backprop)

            # Compute log probabilities for the generated tokens
            labels_for_indexing = generated_tokens[:, 1:].contiguous()
            logits = torch.stack(logits, dim=1)  # Stack logits along the sequence dimension
            
            logits = torch.log_softmax(logits,dim=-1)

            token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

            # Create a mask to ignore padding tokens
            padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
            token_log_probs = token_log_probs * padding_mask.float()

            # Sum log probabilities across the sequence
            sequence_log_prob_sum = token_log_probs.sum(dim=1)

            # Compute policy gradient loss
            val_loss = -(rewards * sequence_log_prob_sum)
            val_loss = val_loss.mean()  # Ensure scalar loss for logging

            print(f"[Validation Step] Policy Gradient Loss: {val_loss}")

        else:
            # ---- MLE LOSS ----
            # Let the T5 model handle the MLE loss internally
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss = outputs.loss  # Use T5's internal cross-entropy loss

            print(f"[Validation Step] MLE Loss: {val_loss}")

        # Log the validation loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations at the end of the validation epoch.
        """
        # Determine CSV path based on test_flag (validation or test)
        csv_file_path = self.test_csv_file_path if test_flag else self.val_csv_file_path

        if self.epoch_validation_details:
            # Log validation details to CSV
            self.log_to_csv(csv_file_path, self.epoch_validation_details)

        # Clean up stored data for the next epoch
        self.cleanup_epoch_data()

    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set, 
        calculate the loss, and optionally generate text.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(test_flag=True)

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

    def cleanup_epoch_data(self):
        """
        Cleans up data collected during the epoch.
        """
        self.epoch_validation_details.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer for the model using AdamW.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
