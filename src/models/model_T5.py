import csv
import logging
import os
import sys
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from src.utils.metrics import MetricsEvaluator
import pytorch_lightning as pl
from pathlib import Path  # Import Path to handle file paths
from src.utils.config import CONFIG
import pandas as pd

logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model on a specific dataset.
    This class handles training, validation, and testing, as well as logging results and managing optimizers.
    """
    def __init__(self, model_name, model_dir):
        """
        Initializes the fine-tuner with the specified model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model to load.
            model_dir (str or Path): Directory where model logs and checkpoints will be saved.
        """
        super().__init__()

        # Ensure model_dir is a Path object for easier file management
        model_dir = Path(model_dir)

        # Load the configuration for the model (with options like attention)
        config = T5Config.from_pretrained(model_name, output_attentions=CONFIG["output_attentions"])

        # Initialize the T5 model and tokenizer with the specified configuration
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Set file paths for saving validation and test details as CSV files
        self.val_csv_file_path = model_dir / "validation_details.csv"
        self.test_csv_file_path = model_dir / "test_details.csv"

        # Initialize lists to store validation step outputs and details for logging purposes
        self.current_val_step_outputs = []
        self.epoch_validation_details = []
  
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, **kwargs):
        """
        Load model from a pre-trained checkpoint and pass additional arguments to the model's __init__ method.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            map_location (optional): Device mapping for loading the checkpoint.
            strict (bool): Whether to strictly enforce that the keys in `state_dict` match the model's keys.
        """
        # Extract model name and model_dir from kwargs (passed during model initialization)
        model_name = kwargs.pop('model_name')
        model_dir = kwargs.pop('model_dir')
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Initialize the model using the provided arguments
        model = cls(model_name=model_name, model_dir=model_dir, **kwargs)
        
        # Load the state_dict (weights) from the checkpoint
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        return model

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model. If labels are provided, it calculates the loss; otherwise, it returns logits.

        Args:
            input_ids (tensor): Input token IDs.
            attention_mask (tensor): Attention mask to avoid attending to padding tokens.
            labels (tensor, optional): Ground truth labels for calculating the loss during training.

        Returns:
            outputs (transformers.modeling_outputs.Seq2SeqLMOutput): The model outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=False  # Set to False, but can be enabled if attention outputs are needed
        )
        return outputs

    def custom_loss(self, outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to token-level loss.

        Args:
            outputs (tensor): Model logits (raw unnormalized scores for each token).
            targets (tensor): Ground truth token IDs.
            differential_weights (tensor): Weights that apply more or less importance to different tokens.

        Returns:
            mean_weighted_loss (float): The mean weighted loss for the batch.
        """
        # Flatten the outputs and targets for token-level comparison
        logits_flat = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten targets to [batch_size * seq_length]
        differential_weights_flat = differential_weights.view(-1)  # Flatten weights similarly

        # Ensure the logits and differential weights have matching shapes
        if logits_flat.size(0) != differential_weights_flat.size(0):
           raise ValueError("Mismatch between logits and differential weights in terms of size.")
        
        # Calculate the cross-entropy loss for each token without reducing to a single value
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply the differential weights to each token's loss
        weighted_loss_per_token = loss_per_token * differential_weights_flat
     
        # Calculate the mean of the weighted losses to get the final batch loss
        mean_weighted_loss = weighted_loss_per_token.mean()
        
        return mean_weighted_loss

    def training_step(self, batch, batch_idx):
        """
        Executes a training step, calculates the loss (including reward-based loss), and logs it.
        """
        # Perform a forward pass to get model outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        # Generate text predictions from the model outputs
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Ground truth (edited endings) from the batch
        edited_endings = batch['edited_ending']

        # Initialize the MetricsEvaluator
        metrics_evaluator = MetricsEvaluator()

        # Calculate reward based on the selected metric in CONFIG
        reward_metric = CONFIG.get("reward_metric", "rouge")
        
        if reward_metric == "rouge":
            rouge_scores = metrics_evaluator.calculate_and_log_rouge_scores(
                generated_texts, edited_endings, None, None, None, None, logger
            )
            reward = rouge_scores.get('rouge_prediction_edited_rouge-l_f', 0) - CONFIG["baseline_score"]
        
        elif reward_metric == "bleu":
            bleu_scores = metrics_evaluator.calculate_and_log_bleu_scores(
                generated_texts, edited_endings, None, None, None, None, logger
            )
            reward = bleu_scores.get('bleu_prediction_edited', 0) - CONFIG["baseline_score"]
        
        elif reward_metric == "bert":
            bert_scores = metrics_evaluator.calculate_and_log_bert_similarity(
                generated_texts, edited_endings, None, None, None, None, logger
            )
            reward = bert_scores.get('bert_prediction_edited_f1', 0) - CONFIG["baseline_score"]
        
        elif reward_metric == "bart":
            bart_scores = metrics_evaluator.calculate_and_log_bart_similarity(
                generated_texts, edited_endings, None, None, None, None, logger
            )
            reward = bart_scores.get('bart_prediction_edited_avg_score', 0) - CONFIG["baseline_score"]

        # Compute the final loss by incorporating the reward into the standard MLE loss
        loss = outputs.loss - CONFIG["reward_weight"] * reward

        # Log the training loss and reward for monitoring
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reward', reward, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        """
        Executes a validation step, calculates validation loss, and logs the results.

        Args:
            batch (dict): A single batch of validation data containing input IDs, attention masks, labels, etc.
            batch_idx (int): The index of the batch in the current validation run.

        Returns:
            None.
        """
        # Perform a forward pass for validation
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        # Calculate validation loss
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Generate text predictions for validation
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Log the details (input and generated output) for further analysis
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

        self.epoch_validation_details.extend(validation_details)

    def generate_text(self, input_ids, attention_mask):
        """
        Generates text from the model's output.

        Args:
            input_ids (tensor): The tokenized input IDs for generation.
            attention_mask (tensor): The attention mask for the input sequence.

        Returns:
            generated_texts (list): A list of generated texts (decoded strings).
        """
        # Generate text sequences with the model, specifying max_length
        generated_ids = self.model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=CONFIG["max_gen_length"],
        )
        # Decode the generated token IDs into text
        generated_texts = [
            self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]
        return generated_texts

    def on_validation_epoch_end(self, test_flag=False):
        """
        Operations performed at the end of each validation epoch.

        Args:
            test_flag (bool): Flag to determine if it's a test run (True) or validation (False).
        """
        # Determine the CSV file path to log validation/test details
        csv_file_path = self.determine_csv_path(test_flag)
        if self.epoch_validation_details:  # Log details if available
            self.log_to_csv(csv_file_path, self.epoch_validation_details)
        else:
            logger.info("No validation details available for logging.")

        # Clear data accumulated over the epoch
        self.cleanup_epoch_data()

    def determine_csv_path(self, test_flag):
        """
        Determines the CSV file path for logging based on test_flag.

        Args:
            test_flag (bool): True for test logging, False for validation logging.

        Returns:
            Path (str): The appropriate file path for saving results.
        """
        return self.test_csv_file_path if test_flag else self.val_csv_file_path

    def log_to_csv(self, csv_file_path, details):
        """
        Logs validation or test details to a CSV file.

        Args:
            csv_file_path (str): Path to the CSV file for logging.
            details (list): List of dictionaries containing the logged information.
        """
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()  # Write header only if the file doesn't exist
            writer.writerows(details)

    def cleanup_epoch_data(self):
        """
        Clears validation or test data stored during the current epoch.
        """
        self.epoch_validation_details.clear()
        self.current_val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop. Same as validation_step.

        Args:
            batch (dict): A single batch of test data.
            batch_idx (int): The index of the batch in the current test run.

        Returns:
            None.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """
        Operations performed at the end of each test epoch.
        """
        return self.on_validation_epoch_end(test_flag=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        The optimizer updates the model's weights to minimize the loss during training.

        Returns:
            optimizer (torch.optim.Optimizer): Optimizer to be used during training.
        """
        lr = CONFIG["learning_rate"]
        return torch.optim.AdamW(self.parameters(), lr=lr)
