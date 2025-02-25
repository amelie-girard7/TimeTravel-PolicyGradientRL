# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/models/model.py
import csv
import logging
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.mle.utils.config import CONFIG
from src.mle.utils.metrics import MetricsEvaluator
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
        # Save only essential hyperparameters
        self.save_hyperparameters('model_name')

        # Store model_dir and file_label as instance variables
        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # Load model and tokenizer
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Set unique file paths using `file_label` to prevent overwriting
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Initialize MetricsEvaluator to handle custom scoring for rewards
        self.metrics_evaluator = MetricsEvaluator()

        # Buffers for validation and testing
        self.epoch_validation_details = []
        self.epoch_test_details = [] 
        self.epoch_scores = []  # Validation scores buffer
        self.epoch_test_scores = [] 

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the T5 model.
        Calculates the loss; otherwise, returns generated tokens.
        """
        # MLE training mode with labels for loss calculation
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=False
        )
        return outputs

    def custom_loss(self, outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to the calculation.
        """
        logits_flat = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten targets to [batch_size * seq_length]
        differential_weights_flat = differential_weights.view(
            -1)  # Flatten weights to match sequence length [batch_size * seq_length]

        # Compute the standard loss function without reduction to get a loss value per token.
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Apply the differential weights to each token's loss.
        weighted_loss_per_token = loss_per_token * differential_weights_flat

        # Calculate the mean of the weighted losses to get a single scalar representing the batch's loss.
        mean_weighted_loss = weighted_loss_per_token.mean()

        return mean_weighted_loss

    def training_step(self, batch, batch_idx):
        """
        Training-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # Forward pass
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_train_loss = outputs.loss

        # Log training loss
        self.log('training_mle_loss', mle_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Decode and log scores for generated texts
        generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_mean = scores.mean()

        self.log('training_mle_score_mean', score_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return mle_train_loss  # Return MLE loss for optimization

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        print(f"Validation Step: Processing batch {batch_idx}")

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_val_loss = outputs.loss

        # Generate predictions
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Calculate sentence-level scores
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        self.epoch_scores.extend(scores.tolist())  # Save validation scores for the dataset

        # Log MLE loss
        self.log('validation_mle_loss', mle_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Save validation details
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })
        return mle_val_loss

    def on_validation_epoch_end(self):
        """
        Finalize and save validation results at the end of the validation epoch.
        """
        print("Validation Epoch End")
        if self.epoch_validation_details:
            print(f"Saving {len(self.epoch_validation_details)} validation details to {self.val_csv_file_path}.")
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)

        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            print(f"Overall validation score: {overall_val_score}")
            self.log("overall_score", overall_val_score, prog_bar=True, logger=True)

        # Clear buffers for next validation run
        self.epoch_validation_details.clear()
        self.epoch_scores.clear()
    
    def test_step(self, batch, batch_idx):
        """
        Test-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_test_loss = outputs.loss

        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        for i in range(len(generated_texts)):
            self.epoch_test_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        return mle_test_loss

    def on_test_epoch_end(self):
        """
        Finalize and save test results at the end of the test epoch.
        """
        print("Test Epoch End")
        if self.epoch_test_details:
            print(f"Saving {len(self.epoch_test_details)} test details to {self.test_csv_file_path}.")
            self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)

        if self.epoch_test_scores:
            overall_test_score = torch.tensor(self.epoch_test_scores).mean().item()
            print(f"Overall test score: {overall_test_score}")
            self.log("test_overall_score", overall_test_score, prog_bar=True, logger=True)

        # Clear buffers for next test run
        self.epoch_test_details.clear()
        self.epoch_test_scores.clear()

    def log_to_csv(self, csv_file_path, details, epoch=None):
        """
        Writes the details to the specified CSV file.
        """
        print(f"Writing {len(details)} entries to {csv_file_path}.")
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()
            for detail in details:
                detail['Epoch'] = epoch
            writer.writerows(details)

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
