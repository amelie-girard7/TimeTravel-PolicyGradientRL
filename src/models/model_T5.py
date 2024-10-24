import csv
import logging
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
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
        
        if labels is not None:
            # If labels are provided and not using policy gradient, the model will calculate MLE loss internally.
            # Otherwise, the model will generate text for policy gradient training or inference.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels, 
                output_attentions=False
            )
            return outputs
        else:
            # If no labels, generate tokens using the model and return logits
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'], 
                num_beams=1,  # Greedy decoding
                output_scores=True,  # Return logits (scores) from the model
                return_dict_in_generate=True  # Return full output (sequences + logits)
            )
            
            generated_tokens = outputs.sequences
            logits = outputs.scores  # Logits for each token generated
                       
            return generated_tokens, logits

    def training_step(self, batch, batch_idx):
        """
        Training step to use either MLE Loss or Policy Gradient Loss based on the config.
        """        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

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
            logits = torch.log_softmax(logits,dim=-1) # Log softmax along the vocabulary axis
            
            token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)
            
            # Create a mask to ignore padding tokens
            padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
            token_log_probs = token_log_probs * padding_mask.float()
            
            # Sum log probabilities across the sequence
            sequence_log_prob_sum = token_log_probs.sum(dim=1)
           
            # Compute policy gradient loss
            loss = -(rewards * sequence_log_prob_sum).mean()

            # Log rewards and loss for training step
            self.log('policy_gradient_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('policy_gradient_reward', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)        
        else:
            # ---- MLE LOSS ----
            # Let the T5 model handle the MLE loss internally
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # T5 internally computes the MLE (cross-entropy) loss
            
            # Log the MLE loss for the training step
            self.log('mle_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, phase="Validation"):
        """
        Validation step to use either MLE Loss or Policy Gradient Loss based on the config.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

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

            # Print the rewards and generated text for debugging
            print(f"Generated texts: {generated_texts}")
            print(f"Rewards: {rewards}")

            # Append validation data
            for i in range(len(generated_texts)):
                validation_data = {
                    "Epoch": self.current_epoch,
                    "Premise": batch['premise'][i],
                    "Initial": batch['initial'][i],
                    "Counterfactual": batch['counterfactual'][i],
                    "Original Ending": batch['original_ending'][i],
                    "Edited Ending": batch['edited_ending'][i],
                    "Generated Text": generated_texts[i],
                    "Reward": rewards[i].item(),
                }
                self.epoch_validation_details.append(validation_data)


            # Compute log probabilities for the generated tokens
            labels_for_indexing = generated_tokens[:, 1:].contiguous()
            # Stack the logits (which are in tuple form) into a tensor
            logits = torch.stack(logits, dim=1)  # Ensure logits are stacked before applying log_softmax

            # Apply log_softmax along the last dimension          
            logits = torch.log_softmax(logits,dim=-1) # Now logits is a single tensor
            
            # Gather token log probabilities based on the generated tokens
            token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

            # Create a mask to ignore padding tokens
            padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
            token_log_probs = token_log_probs * padding_mask.float()

            # Sum log probabilities across the sequence
            sequence_log_prob_sum = token_log_probs.sum(dim=1)

            # Compute policy gradient loss
            loss = -(rewards * sequence_log_prob_sum).mean()

            # Log policy gradient loss and rewards for validation
            self.log(f'validation_policy_gradient_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'validation_policy_gradient_reward', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
      
        else:
            # ---- MLE LOSS ----
            # Let the T5 model handle the MLE loss internally
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Use T5's internal cross-entropy loss

            # Log the loss for the phase
            self.log(f'validation_mle_loss', loss, on_epoch=True, prog_bar=True, logger=True)
   
        return loss

    def log_to_wandb(self, table_name, details):
        """
        Logs the validation or test results into a Weights and Biases (W&B) table.
        """
        # Define the columns for the W&B table
        columns = [
            "Epoch", "Premise", "Initial", "Counterfactual", "Original Ending",
            "Edited Ending", "Generated Text", "Reward"
        ]
        
        # Create a new W&B table with the specified columns
        table = wandb.Table(columns=columns)

        # Add each row of data to the W&B table
        for row in details:
            table.add_data(
                row["Epoch"], row["Premise"], row["Initial"], row["Counterfactual"], 
                row["Original Ending"], row["Edited Ending"], row["Generated Text"], row["Reward"]
            )

        # Log the table to W&B under the correct phase (validation or test)
        wandb.log({f"{table_name}_{self.current_epoch}": table})

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations at the end of the validation or test epoch.
        """
        # Determine the table name based on whether this is validation or test
        table_name = "test_data_epoch" if test_flag else "validation_data_epoch"

        # Only log if there are details
        if self.epoch_validation_details:
            # Log the details to W&B
            self.log_to_wandb(table_name, self.epoch_validation_details)

        # Clear the data after logging
        self.cleanup_epoch_data()

    def test_step(self, batch, batch_idx):
        """
        This method is called during the test loop and should mirror the validation step logic.
        """
        # Use the validation step logic for the test phase
        return self.validation_step(batch, batch_idx, phase="Test")

    def on_test_epoch_end(self):
        """
        Handles operations at the end of the test epoch.
        """
        # Call on_validation_epoch_end with test_flag set to True to log test data
        self.on_validation_epoch_end(test_flag=True)

    def log_to_wandb(self, table_name, details):
        """
        Logs the validation or test results into a Weights and Biases (W&B) table.
        """
        # Define the columns for the W&B table
        columns = [
            "Epoch", "Premise", "Initial", "Counterfactual", "Original Ending",
            "Edited Ending", "Generated Text", "Reward"
        ]
        # Create a new W&B table with the specified columns
        table = wandb.Table(columns=columns)

        # Add each row of validation data to the W&B table
        for row in details:
            table.add_data(
                row["Epoch"], row["Premise"], row["Initial"], row["Counterfactual"], 
                row["Original Ending"], row["Edited Ending"], row["Generated Text"], row["Reward"]
            )

        # Log the table to W&B
        wandb.log({f"{table_name}_{self.current_epoch}": table})

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
