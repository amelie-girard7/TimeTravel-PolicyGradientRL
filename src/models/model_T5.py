import csv
import logging
import os
import sys
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path  # Import Path
from src.utils.config import CONFIG
import pandas as pd

logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model on a specific dataset.
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
  
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, **kwargs):
        """
        Load model from checkpoint and pass additional arguments to the model's __init__ method.
        """
        # Extract model name and model_dir from kwargs
        model_name = kwargs.pop('model_name')
        model_dir = kwargs.pop('model_dir')
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Initialize the model with the provided arguments
        model = cls(model_name=model_name, model_dir=model_dir, **kwargs)
        
        # Load the state_dict from the checkpoint
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        return model

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Performs the forward pass of the model. If labels are provided, it calculates the loss; 
        otherwise, it returns logits. This method also handles the retrieval of attention outputs.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=False  # Ensure attentions are returned True or False
        )
        return outputs

    def custom_loss(self,outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to the calculation.
        
        This function modifies the standard loss function by applying a different
        weight to each token based on its importance, which is determined by the differential_weights tensor.
        This is particularly useful for focusing the model's learning on specific parts of the input data.
        """

        logits_flat = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten targets to [batch_size * seq_length]
        differential_weights_flat = differential_weights.view(-1) # Flatten weights to match sequence length [batch_size * seq_length]  

        # Ensure that the shapes of logits and differential weights align
        if logits_flat.size(0) != differential_weights_flat.size(0):
           raise ValueError("The size of logits and differential weights does not match, indicating a potential issue in preprocessing or batch assembly.")
        
        # Compute the standard loss function without reduction to get a loss value per token.
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        # Apply the differential weights to each token's loss.
        weighted_loss_per_token = loss_per_token * differential_weights_flat
     
        # Calculate the mean of the weighted losses to get a single scalar representing the batch's loss.
        mean_weighted_loss = weighted_loss_per_token.mean()
        
        return mean_weighted_loss

    def training_step(self, batch, batch_idx):
        """
        Executes a training step, calculating the loss and logging it.

        Parameters:
        - batch: A single batch of data containing input IDs, attention masks, and labels.
        - batch_idx: The index of the batch in the current epoch.

        Returns:
        - The loss value for the current batch, used for backpropagation.
        """
        # Perform a forward pass through the model to get the outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        
        # Check the 'use_custom_loss' config and whether 'differential_weights' is in the batch
        if CONFIG['use_custom_loss'] and 'differential_weights' in batch:
            loss = self.custom_loss(outputs.logits, batch['labels'], batch['differential_weights'])
        else:
            # Use the default loss provided by the model outputs
            loss = outputs.loss

            # Log the custom calculated loss for monitoring.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform forward pass
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )

        # Calculate validation loss
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

        # Generate text and capture attentions
        generated_texts = self.generate_text(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        edited_endings = batch['edited_ending']

        # Prepare validation details for logging
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

        # Collect outputs for this validation step
        output = {
            'generated': generated_texts,
            'edited_endings': edited_endings,
            'premises': batch['premise'],
            'counterfactuals': batch['counterfactual'],
            'original_endings': batch['original_ending'],
            'initials': batch['initial'],
        }
        self.current_val_step_outputs.append(output)
        
        # Log average validation loss
        self.log_dict({"avg_val_loss": val_loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

    def generate_text(self, input_ids, attention_mask):
        # Generate text sequences and capture attentions
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

    def on_validation_epoch_end(self, test_flag=False):
        """
        Handles operations to perform at the end of each validation epoch.
        """
        # Handle CSV logging
        csv_file_path = self.determine_csv_path(test_flag)
        if self.epoch_validation_details:  # Check if there are details to log
            self.log_to_csv(csv_file_path, self.epoch_validation_details)
        else:
            logger.info("No validation details available for logging.")

        # Clean up stored data from the current validation epoch
        self.cleanup_epoch_data()
  
    def determine_csv_path(self, test_flag):
        return self.test_csv_file_path if test_flag else self.val_csv_file_path

    def log_to_csv(self, csv_file_path, details):
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(details)

    def cleanup_epoch_data(self):
        self.epoch_validation_details.clear()
        self.current_val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """
        Called during the testing loop to perform a forward pass with a batch from the test set, 
        calculate the loss, and optionally generate text.
        """
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(test_flag=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        The optimizer is responsible for updating the model's weights to minimize the loss during training.
        """
        lr = CONFIG["learning_rate"]
        return torch.optim.AdamW(self.parameters(), lr=lr)
