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
import numpy as np

# Initialize a logger for easier debugging and output control
logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model using policy gradient reinforcement learning.
    This class handles token generation, score calculation, and policy gradient-based optimization.
    """

    def __init__(self, model_name, model_dir):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()
        self.save_hyperparameters()  # Save model_name and model_dir as hyperparameters

        # Ensure model_dir is a Path object for consistent file handling
        model_dir = Path(model_dir)

        # Load the T5 model with specified configurations
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Initialize storage for epoch data
        self.current_epoch_data = []  # Temporary storage for each batch in the current epoch
        self.epoch_validation_details = []  # Final storage of all batches in the last epoch

        # Initialize MetricsEvaluator for score calculation
        self.metrics_evaluator = MetricsEvaluator()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass using the T5 model.
        If labels are provided, it calculates the loss (MLE loss); otherwise, it returns generated tokens and logits.
        """
        if labels is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=False
            )
            return outputs
        else:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],
                num_beams=1,  # Greedy decoding
                output_scores=True,  # Return logits for reward calculation
                return_dict_in_generate=True
            )
            generated_tokens = outputs.sequences
            logits = outputs.scores  # Logits for each token generated
            return generated_tokens, logits

    def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards):
        """
        Helper function to calculate policy gradient loss.
        """
        labels_for_indexing = generated_tokens[:, 1:].contiguous()  # Shift generated tokens to exclude start token
        logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
        token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)
        padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
        token_log_probs = token_log_probs * padding_mask.float()
        sequence_log_prob_sum = token_log_probs.sum(dim=1)

        return -(rewards * sequence_log_prob_sum).mean()

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        if CONFIG["use_policy_gradient"]:
            generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]

            # Calculate scores and adjust by baseline for policy gradient training
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
            rewards = scores - CONFIG["baseline_score"]

            loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards)

            # Log policy gradient metrics for tracking
            self.log('policy_gradient_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('policy_gradient_reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.log('policy_gradient_reward_distribution', wandb.Histogram(rewards.cpu().numpy()), on_epoch=True, logger=True)

            # Print configuration baseline and score
            print(f"Policy Gradient Mode - Baseline: {CONFIG['baseline_score']}, Scores: {scores.tolist()}, Rewards: {rewards.tolist()}")

        else:
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            self.log('mle_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()

            self.log('mle_score_mean', scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            #self.log('mle_score_distribution', wandb.Histogram(scores.cpu().numpy()), on_epoch=True, logger=True)

            print(f"MLE Mode - Scores: {scores.tolist()}")

        return loss

    def validation_step(self, batch, batch_idx, phase="Validation"):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        batch_data = []

        if CONFIG["use_policy_gradient"]:
            generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]

            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
            rewards = scores - CONFIG["baseline_score"]
            loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards)

            self.log(f'validation_policy_gradient_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'validation_policy_gradient_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
            #self.log('validation_policy_gradient_reward_distribution', wandb.Histogram(rewards.cpu().numpy()), on_epoch=True, logger=True)

            batch_data.extend([
                {
                    "Epoch": self.current_epoch,
                    "Premise": batch['premise'][i],
                    "Initial": batch['initial'][i],
                    "Counterfactual": batch['counterfactual'][i],
                    "Original Ending": batch['original_ending'][i],
                    "Edited Ending": batch['edited_ending'][i],
                    "Generated Text": generated_texts[i],
                    "Reward": rewards[i].item(),
                    "Mode": "Policy Gradient"
                } for i in range(len(generated_texts))
            ])

            print(f"Policy Gradient Validation - Baseline: {CONFIG['baseline_score']}, Scores: {scores.tolist()}, Rewards: {rewards.tolist()}")

        else:
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            self.log('validation_mle_loss', loss, on_epoch=True, prog_bar=True, logger=True)

            generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(-1), skip_special_tokens=True)
            edited_endings = [str(ee) for ee in batch['edited_ending']]
            scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()

            self.log('validation_mle_score_mean', scores.mean(), on_epoch=True, prog_bar=True, logger=True)
            #self.log('validation_mle_score_distribution', wandb.Histogram(scores.cpu().numpy()), on_epoch=True, logger=True)

            batch_data.extend([
                {
                    "Epoch": self.current_epoch,
                    "Premise": batch['premise'][i],
                    "Initial": batch['initial'][i],
                    "Counterfactual": batch['counterfactual'][i],
                    "Original Ending": batch['original_ending'][i],
                    "Edited Ending": batch['edited_ending'][i],
                    "Generated Text": generated_texts[i],
                    "Score": scores[i].item(),
                    "Mode": "MLE"
                } for i in range(len(generated_texts))
            ])

            print(f"MLE Validation - Scores: {scores.tolist()}")

        self.current_epoch_data.extend(batch_data)
        return loss

    def on_validation_epoch_end(self, test_flag=False):
        is_last_epoch = (self.current_epoch == (self.trainer.max_epochs - 1))
        
        if is_last_epoch or test_flag:
            table_name = "test_data_epoch" if test_flag else "validation_data_epoch"
            self.epoch_validation_details = self.current_epoch_data
            self.log_to_wandb(table_name, self.epoch_validation_details)
        
        self.current_epoch_data.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, phase="Test")

    def on_test_epoch_end(self):
        # Commented to prevent table logging for test data
        # self.on_validation_epoch_end(test_flag=True)
        pass

    def log_to_wandb(self, table_name, details):
        """
        Logs validation or test results into a Weights and Biases (wandb) table.
        """
        columns = ["Epoch", "Premise", "Initial", "Counterfactual", "Original Ending", 
                "Edited Ending", "Generated Text", "Score/Reward", "Mode"]

        # Create a W&B table with defined columns
        table = wandb.Table(columns=columns)

        # Populate the W&B table with data for each entry in `details`
        for data in details:
            # Use "Reward" for Policy Gradient mode and "Score" for MLE mode
            metric_value = data.get("Reward") if data["Mode"] == "Policy Gradient" else data.get("Score")
            
            # Add row to the W&B table
            table.add_data(
                data["Epoch"], data["Premise"], data["Initial"], data["Counterfactual"],
                data["Original Ending"], data["Edited Ending"], data["Generated Text"],
                metric_value, data["Mode"]
            )

        # Log the table to wandb under a unique name for each epoch
        wandb.log({f"{table_name}_{self.current_epoch}": table})


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
