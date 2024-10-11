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
    The model uses a custom token generation method and computes the loss based on rewards.
    """

    def __init__(self, model_name, model_dir):
        """
        Initializes the fine-tuner with the specified model and tokenizer.

        Args:
            model_name (str): Name of the pre-trained model to load.
            model_dir (str or Path): Directory where model logs and checkpoints will be saved.
        """
        super().__init__()
        # Save the hyperparameters (model_name, model_dir) so that PyTorch Lightning can use them for logging and checkpoints
        self.save_hyperparameters()

        # Load the pre-trained T5 model and the tokenizer from Hugging Face transformers
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # A list to store output during testing, which will be aggregated at the end of the test epoch
        self.test_outputs = []

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, **kwargs):
        """
        Forward pass through the T5 model. This method is typically used during validation and testing.

        Args:
            input_ids (tensor): Encoded input sequences (from tokenizer).
            attention_mask (tensor): Mask to avoid attending to padding tokens.
            decoder_input_ids (tensor): Input to the decoder (previously generated tokens).
            **kwargs: Additional arguments (e.g., past_key_values, use_cache).

        Returns:
            outputs: The model's output (logits, etc.).
        """
        # Forward pass through the T5 model with optional decoder input ids and any additional kwargs
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )

    def custom_generate(self, input_ids, attention_mask, max_length):
        """
        Custom generation function to generate sequences token-by-token, tracking log probabilities 
        for policy gradient computation. This method enables step-wise generation for more control.

        Args:
            input_ids (tensor): Encoded input sequences (from tokenizer).
            attention_mask (tensor): Mask to avoid attending to padding tokens.
            max_length (int): Maximum length for generation.

        Returns:
            generated_ids (tensor): Generated token sequences.
            sequence_log_prob (tensor): Log probabilities for the entire sequence.
        """
        batch_size = input_ids.size(0)  # The batch size, i.e., how many sequences we are processing at once

        # The first token input to the decoder is always the "start" token for T5.
        decoder_start_token_id = self.model.config.decoder_start_token_id

        # We initialize decoder inputs with the start token for all sequences in the batch
        decoder_input_ids = torch.full(
            (batch_size, 1), decoder_start_token_id, dtype=torch.long, device=input_ids.device
        )

        # We initialize `generated_ids` with the decoder start token (first token) for all sequences
        generated_ids = decoder_input_ids
        
        # `log_probs` will store the log probability of each token generated
        log_probs = []

        # `past_key_values` helps the model remember the past token sequence for more efficient computation
        past_key_values = None

        # Generate tokens one-by-one, iterating for up to `max_length`
        for t in range(max_length):
            # Forward pass through the model with the current decoder input and previous tokens (cached in past_key_values)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                past_key_values=past_key_values,
                use_cache=True,  # Enable caching of previous key/values for faster generation
            )

            # The logits (unnormalized probabilities) for the next token are the last token's output from the model
            next_token_logits = outputs.logits[:, -1, :]
            
            # Cache the past key/values for efficient sequential generation
            past_key_values = outputs.past_key_values

            # Compute probabilities from the logits and sample a token from the probability distribution
            probs = torch.softmax(next_token_logits, dim=-1)  # Softmax to get probabilities
            m = torch.distributions.Categorical(probs)  # Treat these probabilities as a categorical distribution

            next_tokens = m.sample()  # Sample the next token from the distribution
            selected_log_probs = m.log_prob(next_tokens)  # Get the log probability of the sampled token
            log_probs.append(selected_log_probs)  # Store the log probability

            # Append the new token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Update the decoder inputs with the newly generated token for the next iteration
            decoder_input_ids = next_tokens.unsqueeze(-1)

        # After generating all tokens, we stack the log probabilities for each token and sum them across the sequence
        sequence_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)  # Shape: (batch_size,)
        
        return generated_ids, sequence_log_prob

    def training_step(self, batch, batch_idx):
        """
        Training step for policy gradient-based fine-tuning. The model generates sequences,
        computes rewards, and updates the model based on policy gradient loss.

        Args:
            batch (dict): The input batch of data.
            batch_idx (int): The batch index.

        Returns:
            loss (tensor): The computed loss value for the current batch.
        """
        input_ids = batch['input_ids']  # The input sequences (tokenized)
        attention_mask = batch['attention_mask']  # Mask to prevent attending to padding tokens
        batch_size = input_ids.size(0)

        # Debugging: Check the shape of input tensors to ensure correctness
        print(f"DEBUG: input_ids shape = {input_ids.shape}")
        print(f"DEBUG: attention_mask shape = {attention_mask.shape}")

        # Use custom_generate to handle token generation and log probability computation
        generated_ids, sequence_log_prob = self.custom_generate(
            input_ids, attention_mask, CONFIG["max_gen_length"]  # CONFIG defines the max sequence length
        )

        # Shift the generated_ids to remove the first token (the start token), which isn't part of the final output
        generated_ids_shifted = generated_ids[:, 1:]

        # Decode the generated token IDs back to text for comparison with ground truth (reference endings)
        generated_texts = self.tokenizer.batch_decode(
            generated_ids_shifted, skip_special_tokens=True
        )

        # Debugging: Check the generated text sequences
        print(f"DEBUG: Generated texts: {generated_texts}")

        # Get the reference endings (ground truth) from the batch
        edited_endings = batch['edited_ending']
        
        # If the ground truth is a tuple, convert it to a list (to handle various input formats)
        if isinstance(edited_endings, tuple):
            edited_endings = list(edited_endings)
        
        # Convert the ground truth to string format for comparison
        edited_endings = [str(ee) for ee in edited_endings]

        # Debugging: Check the reference (ground truth) edited endings
        print(f"DEBUG: Reference edited endings: {edited_endings}")

        # Use the MetricsEvaluator class to compute the reward based on generated texts and ground truth
        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)
        
        # Convert rewards into a tensor for use in the loss computation
        rewards = torch.tensor(rewards, device=sequence_log_prob.device)
        
        # Subtract the baseline score from the rewards for normalization (as per policy gradient method)
        rewards = rewards - CONFIG["baseline_score"]

        # Debugging: Check the computed rewards
        print(f"DEBUG: Rewards: {rewards}")

        # Compute the policy gradient loss as negative rewards multiplied by log probabilities of generated tokens
        loss = -rewards * sequence_log_prob
        loss = loss.mean()  # Take the mean loss over the batch

        # Log the training loss and the average reward to TensorBoard
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
        # Forward pass through the model using the input_ids and attention_mask
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Extract the validation loss from the model's output
        val_loss = outputs.loss

        # Log the validation loss to track performance during training
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

        # Generate output sequences using beam search (or greedy search) for testing
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=CONFIG["max_gen_length"],
            num_beams=CONFIG.get("num_beams", 5),  # Use beam search if configured
            early_stopping=True  # Stop generation when all beams are finished
        )

        # Decode the generated sequences to human-readable text
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Get the reference edited endings from the batch
        edited_endings = batch['edited_ending']

        # If the edited endings are a tuple, convert them to a list
        if isinstance(edited_endings, tuple):
            edited_endings = list(edited_endings)
        
        # Convert the edited endings to string format for comparison
        edited_endings = [str(ee) for ee in edited_endings]

        # Debugging: Check the generated texts and the ground truth edited endings
        print("DEBUG (Test Step): generated_texts:", generated_texts)
        print("DEBUG (Test Step): edited_endings:", edited_endings)

        # Use the metrics evaluator to calculate rewards (e.g., ROUGE scores)
        metrics_evaluator = MetricsEvaluator()
        rewards = metrics_evaluator.calculate_reward(generated_texts, edited_endings)

        # Store the rewards for aggregation at the end of the test epoch
        self.test_outputs.extend(rewards)

        # Compute and log the average reward for this batch
        avg_reward = torch.tensor(rewards).mean()
        self.log('test_reward', avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to aggregate and log metrics.

        Returns:
            None
        """
        # Aggregate all rewards collected during the test epoch
        avg_test_reward = torch.tensor(self.test_outputs).mean()

        # Log the aggregated reward as the final test metric
        self.log('avg_test_reward', avg_test_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Clear the test outputs list for the next test run
        self.test_outputs = []

    def configure_optimizers(self):
        """
        Configures the optimizer for training. Uses AdamW with the learning rate specified in CONFIG.

        Returns:
            optimizer (torch.optim.Optimizer): The AdamW optimizer.
        """
        # Get the learning rate from the CONFIG dictionary
        lr = CONFIG["learning_rate"]

        # Use AdamW optimizer, which is well-suited for transformer-based models
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        return optimizer
