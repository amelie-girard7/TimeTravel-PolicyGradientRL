import csv
import logging
import os
import torch
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.metrics import MetricsEvaluator

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

        # Load the configuration for the model with output_attentions
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"],
            use_cache=False  # Important for gradient flow
        )

        # Initialize the T5 model and tokenizer with the specified configuration
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Set file paths for saving validation and test details as CSV files
        self.val_csv_file_path = model_dir / "validation_details.csv"
        self.test_csv_file_path = model_dir / "test_details.csv"

        # Initialize lists to store validation details
        self.current_val_step_outputs = []
        self.epoch_validation_details = []

    def forward(self, input_ids, attention_mask):
        # Generate sequences with gradient tracking and obtain scores
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["max_gen_length"],
            num_beams=1,
            do_sample=True,  # Ensure randomness for exploration
            output_scores=True,               # Return scores (logits)
            return_dict_in_generate=True,     # Return full output including logits
            use_cache=False,                  # Important for gradient flow
            output_attentions=CONFIG["output_attentions"],
            output_hidden_states=True,        # Include hidden states for gradient flow
        )
        return outputs  # Returns a dict with 'sequences' and 'scores'
  
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Step 1: Generate sequences and obtain scores
        outputs = self.forward(input_ids, attention_mask)
        generated_ids = outputs.sequences  # Shape: [batch_size, seq_len_generated]
        scores = outputs.scores  # List of tensors, each shape [batch_size, vocab_size]

        # Verify that scores require gradients
        for idx, score in enumerate(scores):
            if not score.requires_grad:
                score.retain_grad()

        # Step 2: Compute log probabilities of the generated tokens
        token_log_probs = []
        for idx, score in enumerate(scores):
            # Compute log_softmax over the vocabulary for time step idx
            log_probs = torch.nn.functional.log_softmax(score, dim=-1)  # Shape: [batch_size, vocab_size]

            # Get the token IDs generated at time step idx+1
            # (since scores correspond to tokens generated after the current one)
            token_ids = generated_ids[:, idx + 1]  # Shape: [batch_size]

            # Gather log probabilities of the generated tokens
            token_log_prob = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
            token_log_probs.append(token_log_prob)

        # Stack and sum log probabilities
        token_log_probs = torch.stack(token_log_probs, dim=1)  # Shape: [batch_size, seq_len_generated - 1]
        sequence_log_probs = token_log_probs.sum(dim=1)        # Shape: [batch_size]

        # Step 3: Calculate rewards based on generated texts
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
        rewards = rewards - CONFIG["baseline_score"]
        rewards = rewards.detach()  # Detach rewards as they are not differentiable

        # Step 4: Compute the policy gradient loss
        loss = -torch.mean(rewards * sequence_log_probs)

        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reward', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step and logs evaluation metrics.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Generate sequences using self.forward()
        outputs = self.forward(input_ids, attention_mask)
        generated_ids = outputs.sequences
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Compute evaluation metrics
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
        avg_reward = rewards.mean().item()

        # Log the average reward
        self.log('val_avg_reward', avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Collect validation details
        validation_details = [{
            'Epoch': self.current_epoch,
            'Premise': premise,
            'Initial': initial,
            'Counterfactual': counterfactual,
            'Original Ending': original_ending,
            'Edited Ending': edited_ending,
            'Generated Text': generated_text,
            'Reward': reward.item(),
        } for premise, initial, counterfactual, original_ending, edited_ending, generated_text, reward
        in zip(batch['premise'], batch['initial'], batch['counterfactual'],
               batch['original_ending'], batch['edited_ending'], generated_texts, rewards)]

        self.epoch_validation_details.extend(validation_details)

        # Debugging information
        print(f"Validation step completed for batch {batch_idx}")
        print(f"Validation details: {validation_details}")

    def on_validation_epoch_end(self):
        """
        Handles operations at the end of each validation epoch.
        """
        # Determine CSV path
        csv_file_path = self.val_csv_file_path
        
        # Log validation details to CSV if available
        if self.epoch_validation_details:
            print(f"Logging validation details to {csv_file_path}")
            self.log_to_csv(csv_file_path, self.epoch_validation_details)
        else:
            logger.info("No validation details available for logging.")
            print("No validation details available for logging.")

        # Clean up for the next epoch
        self.cleanup_epoch_data()

    def test_step(self, batch, batch_idx):
        """
        Executes a test step by reusing validation logic.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """
        Handles the end of the test epoch by calling the validation end logic.
        """
        return self.on_validation_epoch_end()

    def determine_csv_path(self, test_flag):
        """
        Determines the CSV file path based on whether this is a validation or test step.
        """
        return self.test_csv_file_path if test_flag else self.val_csv_file_path

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
        Cleans up the data collected during the epoch.
        """
        self.epoch_validation_details.clear()
        self.current_val_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer for the model using AdamW.
        """
        lr = CONFIG["learning_rate"]
        return torch.optim.AdamW(self.parameters(), lr=lr)
