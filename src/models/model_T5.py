# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/models/model_T5.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.metrics import MetricsEvaluator
from src.utils.config import CONFIG
import torch.distributions

class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model with policy gradient methods.
    """

    def __init__(self, model_name, model_dir):
        """
        Initializes the fine-tuner with the specified model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model to load.
            model_dir (str or Path): Directory where model logs and checkpoints will be saved.
        """
        super().__init__()
        # Save hyperparameters before assigning model attributes
        self.save_hyperparameters()
        # Load the pre-trained T5 model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Initialize a list to store test outputs
        self.test_outputs = []

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, **kwargs):
        """
        Forward pass through the T5 model.

        Args:
            input_ids (tensor): Encoded input sequences (from tokenizer).
            attention_mask (tensor): Mask to avoid attending to padding tokens.
            decoder_input_ids (tensor): Input to the decoder (previously generated tokens).
            **kwargs: Additional arguments (e.g., past_key_values, use_cache).

        Returns:
            outputs: The model's output (logits, etc.).
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step, computing the loss and reward.

        Args:
            batch (dict): A batch of input data.
            batch_idx (int): Index of the batch (for tracking during training).

        Returns:
            loss (tensor): The calculated policy gradient loss for the batch.
        """
        # [Your existing training_step code with debugging prints]
        # ... (omitted for brevity)
        pass  # Replace this line with your actual training_step code

    def validation_step(self, batch, batch_idx):
        """
        Validation step to compute the validation loss.

        Args:
            batch (dict): A batch of validation data.

        Returns:
            val_loss (tensor): The validation loss.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Test step to compute metrics during testing.

        Args:
            batch (dict): A batch of test data.
            batch_idx (int): Index of the batch (for tracking during testing).

        Returns:
            None
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.size(0)

        # Generate outputs using the model's generate method
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["max_gen_length"],
            num_beams=CONFIG.get("num_beams", 5),
            early_stopping=True
        )

        # Decode the generated token IDs back to text
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Get the reference endings
        edited_endings = batch['edited_ending']  # The reference (ground truth) endings

        # Ensure edited_endings is a list of strings
        if isinstance(edited_endings, tuple):
            edited_endings = list(edited_endings)
        edited_endings = [str(ee) for ee in edited_endings]

        # Debugging prints
        print("DEBUG (Test Step): Type of generated_texts:", type(generated_texts))
        print("DEBUG (Test Step): generated_texts:", generated_texts)
        print("DEBUG (Test Step): Type of edited_endings:", type(edited_endings))
        print("DEBUG (Test Step): edited_endings:", edited_endings)

        # Compute metrics using the MetricsEvaluator class
        metrics_evaluator = MetricsEvaluator()

        # Calculate rewards (e.g., ROUGE-L F1 scores)
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)

        # Store outputs for epoch end processing
        self.test_outputs.extend(rewards)

        # Log the metrics
        avg_reward = torch.tensor(rewards).mean()
        self.log('test_reward', avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to aggregate and log metrics.

        Returns:
            None
        """
        # Aggregate metrics
        avg_test_reward = torch.tensor(self.test_outputs).mean()

        # Log aggregated metrics
        self.log('avg_test_reward', avg_test_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear the outputs for the next test run
        self.test_outputs = []

    def configure_optimizers(self):
        """
        Configures the optimizer for training. Uses AdamW with the learning rate specified in CONFIG.

        Returns:
            optimizer (torch.optim.Optimizer): The AdamW optimizer.
        """
        lr = CONFIG["learning_rate"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return optimizer
