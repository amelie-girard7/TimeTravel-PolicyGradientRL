import time
import json
import logging
import pandas as pd
import torch
import torch.nn.utils.rnn
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

def count_json_lines(file_path):
    """
    Counts the number of lines in a JSON file, useful for dataset size estimation.
    """
    logger.info(f"Counting lines in file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return sum(1 for _ in file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

def load_first_line_from_json(file_path):
    """
    Loads and parses the first line from a JSON file for quick data inspection.
    """
    logger.info(f"Loading first line from JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.loads(next(file))
    except Exception as e:
        logger.error(f"Error reading from {file_path}: {e}")
        raise IOError(f"Error reading from {file_path}: {e}")

def preprocess_data(row, tokenizer):
    """
    Prepares a single row of data for model input by tokenizing the text fields.
    Constructs the input sequence by combining story parts and tokenizes them.
    """
    logger.debug("Preprocessing data row...")

    try:
        # Define the separator token specific to the T5 model
        separator_token = "</s>"

        # Combine the premise, initial event, and original ending to create the input sequence
        input_sequence = (
            f"{row['premise']}"
            f"{row['initial']}"
            f"{row['original_ending']} {separator_token} "
            f"{row['premise']} {row['counterfactual']}"
        )

        # Tokenize the input sequence with truncation to max_length and no padding
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        # Tokenize the edited ending, which serves as the target sequence
        tokenized_ending = tokenizer.encode_plus(
            row['edited_ending'], truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        logger.debug(f"Input IDs: {tokenized_inputs['input_ids']}")
        logger.debug(f"Attention Mask: {tokenized_inputs['attention_mask']}")
        logger.debug(f"Labels: {tokenized_ending['input_ids']}")

        # Return the tokenized inputs and labels for training
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': tokenized_ending['input_ids'].squeeze(0),
            # Additional fields for evaluation
            'premise': row['premise'],
            'initial': row['initial'],
            'original_ending': row['original_ending'],
            'counterfactual': row['counterfactual'],
            'edited_ending': row['edited_ending']
        }

    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None

def collate_fn(batch, pad_token_id=0, attention_pad_value=0):
    """
    Collates a batch of preprocessed data into a format suitable for model input,
    applying padding to equalize sequence lengths within the batch.
    """
    # Unpack the batch into separate lists for each field
    input_ids, attention_mask, labels, premise, initial, original_ending, counterfactual, edited_ending = list(zip(*batch))

    # Pad the sequences for 'input_ids', 'attention_masks', and 'labels'
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=attention_pad_value)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    # Return the padded tensors and other fields for further processing
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded,
        'premise': premise,
        'initial': initial,
        'original_ending': original_ending,
        'counterfactual': counterfactual,
        'edited_ending': edited_ending,
    }
