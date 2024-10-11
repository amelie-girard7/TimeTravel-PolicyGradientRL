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
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.size(0)

        # Initialize the decoder input with the decoder start token
        decoder_start_token_id = self.model.config.decoder_start_token_id
        decoder_input_ids = torch.full(
            (batch_size, 1), decoder_start_token_id, dtype=torch.long, device=input_ids.device
        )
        generated_ids = decoder_input_ids
        log_probs = []
        max_length = CONFIG["max_gen_length"]
        past_key_values = None

        for t in range(max_length):
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            probs = torch.softmax(next_token_logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            next_tokens = m.sample()
            selected_log_probs = m.log_prob(next_tokens)
            log_probs.append(selected_log_probs)

            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            decoder_input_ids = next_tokens.unsqueeze(-1)

        sequence_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)

        # Decode the generated token IDs back to text
        generated_texts = self.tokenizer.batch_decode(
            generated_ids[:, 1:], skip_special_tokens=True
        )

        # Get the reference endings
        edited_endings = batch['edited_ending']
        if isinstance(edited_endings, tuple):
            edited_endings = list(edited_endings)
        edited_endings = [str(ee) for ee in edited_endings]

        # Compute the reward
        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
        rewards = torch.tensor(rewards, device=sequence_log_prob.device)
        rewards = rewards - CONFIG["baseline_score"]

        # Compute the loss
        loss = -rewards * sequence_log_prob
        loss = loss.mean()

        # Log the training loss and reward
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_reward', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

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
