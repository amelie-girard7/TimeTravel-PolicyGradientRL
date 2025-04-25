# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/mle/models/model_T5.py
import csv
import logging
import os
import sys
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path  # Import Path
from src.mle.utils.config import CONFIG
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

        # Load the configuration for the model 
        config = T5Config.from_pretrained(model_name)

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

        # Add value head for PPO compatibility
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.d_model, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        
        # Initialize value head properly
        self._initialize_value_head()

    def _initialize_value_head(self):
        """Standardized initialization matching MLE model"""
        try:
            # First try loading from checkpoint if specified
            if CONFIG.get("init_value_head") == "from_mle" and CONFIG.get("ppo_checkpoint_path"):
                checkpoint = torch.load(CONFIG["ppo_checkpoint_path"], map_location='cpu')
                if 'value_head_state_dict' in checkpoint:
                    self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
                    logger.info("Loaded value head from MLE checkpoint")
                    return
            
            # Fallback to model-based initialization
            with torch.no_grad():
                decoder_weights = self.model.decoder.block[-1].layer[2].DenseReluDense.wi_0.weight
                self.value_head[0].weight.data.copy_(decoder_weights[:512])
                self.value_head[2].weight.data.copy_(decoder_weights[:1, :512].t())
                
                if hasattr(self.value_head[0], 'bias'):
                    self.value_head[0].bias.data.zero_()
                if hasattr(self.value_head[2], 'bias'):
                    self.value_head[2].bias.data.zero_()
        except Exception as e:
            logger.error(f"Value head initialization failed: {e}")
            # Final fallback
            for layer in self.value_head:
                if hasattr(layer, 'weight'):
                    torch.nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias'):
                    layer.bias.data.zero_()
            
    def on_save_checkpoint(self, checkpoint):
        checkpoint.update({
            'value_head_state_dict': dict(self.value_head.state_dict()),
            'model_config': self.model.config.to_dict(),
            'checkpoint_type': 'mle',
            'git_hash': os.popen('git rev-parse HEAD').read().strip()
        })

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, **kwargs):
        """More robust checkpoint loading"""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Handle both old and new checkpoint formats
        if 'model_config' in checkpoint:
            model = cls(
                model_name=kwargs.pop('model_name'),
                model_dir=kwargs.pop('model_dir'),
                **kwargs
            )
        else:
            # Fallback for old checkpoints
            model = cls(
                model_name=checkpoint.get('hyperparameters', {}).get('model_name'),
                model_dir=checkpoint.get('hyperparameters', {}).get('model_dir'),
                **kwargs
            )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        # Load value head if available
        if 'value_head_state_dict' in checkpoint:
            model.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        else:
            logger.warning("No value head found in checkpoint, initializing new one")
            model._initialize_value_head()
            
        return model

    def forward(self, input_ids, labels=None):
        """
        Performs the forward pass of the model. If labels are provided, it calculates the loss; 
        otherwise, it returns logits. This method 
        """

        """Add shape validation"""
        assert input_ids.dim() == 2, f"input_ids should be 2D [batch, seq], got {input_ids.shape}"
        if labels is not None:
            assert labels.dim() == 2, f"labels should be 2D [batch, seq], got {labels.shape}"
    
        outputs = self.model(
            input_ids=input_ids,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Executes a training step, calculating the loss and logging it.

        Parameters:
        - batch: A single batch of data containing input IDs,  and labels.
        - batch_idx: The index of the batch in the current epoch.

        Returns:
        - The loss value for the current batch, used for backpropagation.
        """
        # Perform a forward pass through the model to get the outputs
        outputs = self.forward(
            input_ids=batch['input_ids'],
            labels=batch['labels']
        )
        
        loss = outputs.loss

            # Log the custom calculated loss for monitoring.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Perform forward pass
        outputs = self.forward(
            input_ids=batch['input_ids'],
            labels=batch['labels']
        )

        # Calculate validation loss
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['input_ids'].size(0))

        # Generate text 
        generated_texts = self.generate_text(
            input_ids=batch['input_ids']
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

    def generate_text(self, input_ids):
        """Generate text with proper configuration for PPO training"""
        assert input_ids.dim() == 2, f"input_ids should be 2D [batch, seq], got {input_ids.shape}"
        # Generate with sampling for diversity (important for PPO)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_length=CONFIG["max_gen_length"],
            do_sample=True,  # Match PPO behavior
            temperature=CONFIG.get("temperature", 0.7),
            top_k=CONFIG.get("top_k", 50),
            top_p=CONFIG.get("top_p", 0.9),
            num_return_sequences=1
        )
            
        # Handle batch dimension properly
        if generated_ids.dim() == 1:  # Single sequence case
            generated_ids = generated_ids.unsqueeze(0)
        
        # Keep this critical decoding step!
        generated_texts = [
            self.tokenizer.decode(
                generated_id, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            for generated_id in generated_ids
        ]
        
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
