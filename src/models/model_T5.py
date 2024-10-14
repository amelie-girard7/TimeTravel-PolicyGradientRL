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
  
    def forward(self, input_ids, attention_mask, decoder_input_ids=None, **kwargs):
        """
        Forward pass using the T5 model. This method uses the Hugging Face `generate` function
        to generate token sequences and log probabilities (for reinforcement learning).
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=kwargs.get('max_length', 50),  # Default max_length can be 50 or CONFIG-based
            num_beams=1,  # Ensuring one sequence per input
            output_scores=True,  # Return scores (logits)
            return_dict_in_generate=True  # Return the full output including scores and sequences
        )

        # Log probabilities of the generated tokens for policy gradient computation
        log_probs = torch.log_softmax(outputs.scores[-1], dim=-1)

        # Return the generated sequences and log probabilities
        return outputs['sequences'], log_probs

    def training_step(self, batch, batch_idx):
        """
        Training step for policy gradient-based fine-tuning.
        The model generates sequences, computes rewards, and calculates policy gradient loss.
        
        Args:
            batch (dict): A batch of input data.
            batch_idx (int): The index of the current batch.

        Returns:
            tensor: The computed loss value for the current training batch.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Print the batch size of input
        print(f"Batch size (input_ids): {input_ids.size(0)}")

        # Call the forward function to generate token sequences and log probabilities.
        generated_ids, log_probs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=CONFIG["max_gen_length"]
        )

        # Print the sizes of generated sequences and log probabilities
        print(f"Generated sequences size: {generated_ids.size()}")  # Should be (batch_size, seq_len)
        print(f"Log probabilities size: {log_probs.size()}")  # Should match the number of tokens per sequence

        # Shift generated sequences to remove the start token for decoding.
        generated_ids_shifted = generated_ids[:, 1:]

        # Decode the generated token IDs into text for comparison with ground truth.
        generated_texts = self.tokenizer.batch_decode(generated_ids_shifted, skip_special_tokens=True)

        # Ground truth edited endings for reward calculation.
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Print the number of generated texts and edited endings
        print(f"Number of generated texts: {len(generated_texts)}")
        print(f"Number of edited endings: {len(edited_endings)}")

        # Use the MetricsEvaluator to calculate the reward (e.g., based on ROUGE or BERTScore).
        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)

        # Print the size of rewards
        print(f"Rewards size: {len(rewards)}")  # Should match the number of sequences

        # Convert rewards to a tensor (without gradient tracking) and move to the correct device (GPU 0).
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda:0')

        # Ensure that the rewards match the batch size
        if rewards.size(0) != input_ids.size(0):
            raise ValueError(f"Reward size ({rewards.size(0)}) does not match batch size ({input_ids.size(0)}).")

        # Subtract the baseline score to normalize rewards for policy gradient calculation.
        rewards = rewards - CONFIG["baseline_score"]

        # Sum the log probabilities over the token dimension to get sequence-level log probabilities
        if isinstance(log_probs, list):
            # Each element in log_probs corresponds to token-level log probabilities
            # Summing over the tokens for each sequence to get sequence-level log_probs
            sequence_log_prob_sum = torch.stack([lp.sum(dim=-1) for lp in log_probs], dim=0).sum(dim=-1)
        else:
            # Direct sum of the log_probs if it was returned in another format
            sequence_log_prob_sum = log_probs.sum(dim=1)  # Summing over the token (time) dimension

        # Print the size of the summed log probabilities
        print(f"Summed log probabilities size: {sequence_log_prob_sum.size()}")  # Should match the batch size

        # Ensure that the sizes of rewards and sequence_log_prob_sum match before computing the loss
        if rewards.size(0) != sequence_log_prob_sum.size(0):
            raise ValueError(f"Mismatch in rewards and log_probs sizes: rewards ({rewards.size(0)}), log_probs ({sequence_log_prob_sum.size(0)})")

        # Compute the policy gradient loss as the negative rewards times the log probabilities.
        loss = -rewards * sequence_log_prob_sum

        # Ensure that loss requires a gradient
        loss = loss.mean().requires_grad_()  # Ensure loss requires gradient computation

        # Print the final loss value
        print(f"Training loss: {loss.item()}")

        # Log the training loss and average reward.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reward', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step and logs validation metrics.

        Args:
            batch (dict): A batch of validation data.

        Returns:
            tensor: Validation loss.
        """
        # Forward pass through the T5 model using input_ids and attention_mask.
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Calculate the output logits and validation loss.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss

        # Log the validation loss.
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text from the model for comparison with edited endings.
        generated_texts = self.generate_text(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

        # Ground truth edited endings for evaluation
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Append validation details for CSV logging later
        validation_details = [{
            'Epoch': self.current_epoch,
            'Premise': premise,
            'Initial': initial,
            'Counterfactual': counterfactual,
            'Original Ending': original_ending,
            'Edited Ending': edited_ending,
            'Generated Text': generated_text,
        } for premise, initial, counterfactual, original_ending, edited_ending, generated_text
        in zip(batch['premise'], batch['initial'], batch['counterfactual'], batch['original_ending'], batch['edited_ending'], generated_texts)]

        # Collect validation details to log at the end of the epoch
        self.epoch_validation_details.extend(validation_details)

        # Print for debugging purposes
        print(f"Validation step completed for batch {batch_idx}")
        print(f"Validation details: {validation_details}")

        return val_loss

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
