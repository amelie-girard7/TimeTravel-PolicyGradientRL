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
        config = T5Config.from_pretrained(model_name, output_attentions=CONFIG["output_attentions"])

        # Initialize the T5 model and tokenizer with the specified configuration
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Set file paths for saving validation and test details as CSV files
        self.val_csv_file_path = model_dir / "validation_details.csv"
        self.test_csv_file_path = model_dir / "test_details.csv"

        # Initialize the list to store validation step outputs for aggregating results over an epoch
        self.current_val_step_outputs = []
        
        # Initialize a list to store detailed validation information for logging purposes
        self.epoch_validation_details = []

    def forward(self, input_ids, attention_mask):
        # Generate sequences with gradient tracking
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["max_gen_length"],
            num_beams=1,
            do_sample=True,  # Ensure randomness for exploration
            output_scores=False,
            return_dict_in_generate=False,
        )
        return outputs  # Returns generated_ids
  
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        device = input_ids.device

        # Step 1: Generate sequences (y) with gradient tracking
        generated_ids = self.forward(input_ids, attention_mask)
        batch_size = input_ids.size(0)

        # Decode generated sequences into text for reward calculation
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Step 2: Calculate rewards based on generated texts
        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
        rewards = rewards - CONFIG["baseline_score"]
        rewards = rewards.to(device)
        rewards = rewards.detach()  # Detach rewards as they are not differentiable

        # Step 3: Compute log probabilities of the generated sequences (log_prob(y))
        # Prepare labels and decoder inputs by shifting generated_ids
        labels = generated_ids[:, 1:].contiguous()
        decoder_input_ids = generated_ids[:, :-1].contiguous()

        # Forward pass to get logits connected to the computation graph
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

        # Compute log probabilities over the vocabulary
        log_probs = torch.log_softmax(logits, dim=-1)

        # Step 4: Gather log probabilities corresponding to generated tokens
        vocab_size = logits.size(-1)
        labels_for_indexing = labels.clone()
        labels_for_indexing[labels_for_indexing >= vocab_size] = 0
        labels_for_indexing[labels_for_indexing < 0] = 0

        # Gather the log probabilities of the generated tokens
        token_log_probs = log_probs.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

        # Create a mask to ignore padding tokens in the loss computation
        padding_mask = labels != self.tokenizer.pad_token_id

        # Apply the padding mask to token_log_probs
        token_log_probs = token_log_probs * padding_mask.float()

        # Sum log probabilities over the sequence length to get log_prob(y)
        sequence_log_probs = token_log_probs.sum(dim=1)

        # Step 5: Compute the policy gradient loss
        loss = -torch.mean(rewards * sequence_log_probs)

        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reward', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step and logs validation metrics.

        Args:
            batch (dict): A batch of validation data.

        Returns:
            tensor: Validation loss (optional).
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        device = input_ids.device

        # Generate sequences using self.forward()
        generated_ids = self.forward(input_ids, attention_mask)

        # Decode generated sequences into text
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # (Optional) Calculate validation loss using the same method as training
        # Note: Since we don't have rewards in validation, you might skip loss computation
        val_loss = None  # Or compute if necessary

        # Collect validation details for logging
        validation_details = [{
            'Epoch': self.current_epoch,
            'Premise': premise,
            'Initial': initial,
            'Counterfactual': counterfactual,
            'Original Ending': original_ending,
            'Edited Ending': edited_ending,
            'Generated Text': generated_text,
        } for premise, initial, counterfactual, original_ending, edited_ending, generated_text
        in zip(batch['premise'], batch['initial'], batch['counterfactual'],
            batch['original_ending'], batch['edited_ending'], generated_texts)]

        self.epoch_validation_details.extend(validation_details)

        # Debugging information
        print(f"Validation step completed for batch {batch_idx}")
        print(f"Validation details: {validation_details}")

        # You can log metrics if needed
        # self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_loss  # Or return None if not computing loss

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations at the end of each validation epoch.
        Saves the results to CSV and cleans up temporary data.
        """
        print(f"Epoch validation details: {self.epoch_validation_details}")  # Debug print

        # Determine CSV path based on test_flag (whether this is validation or test)
        csv_file_path = self.determine_csv_path(test_flag)
        
        # Log validation details to CSV if available
        if self.epoch_validation_details:
            print(f"Logging validation details to {csv_file_path}")
            self.log_to_csv(csv_file_path, self.epoch_validation_details)
        else:
            logger.info("No validation details available for logging.")
            print("No validation details available for logging.")

        # Clean up stored data for the next epoch
        self.cleanup_epoch_data()

    def test_step(self, batch, batch_idx):
        """
        Executes a test step by reusing validation logic.

        Args:
            batch (dict): A batch of test data.
            batch_idx (int): Index of the batch.

        Returns:
            tensor: Validation (test) loss.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """
        Handles the end of the test epoch by calling the validation end logic and saving test results.
        """
        return self.on_validation_epoch_end(test_flag=True)

    def generate_text(self, input_ids, attention_mask):
        """
        Generates text sequences based on input_ids and attention_mask.
        """
        # Generate text sequences
        generated_ids = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=CONFIG["max_gen_length"],
        )
        # Decode generated sequences into text
        generated_texts = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        # Return generated texts 
        return generated_texts

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
