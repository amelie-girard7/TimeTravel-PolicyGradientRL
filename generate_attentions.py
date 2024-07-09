import os  # Import the os module
import torch
import json
import numpy as np
import pandas as pd
from transformers import T5Tokenizer
from pathlib import Path
from src.models.model_T5 import FlanT5FineTuner  # Ensure this is the correct import path
from src.utils.config import CONFIG  # Ensure this is the correct import path

# Define the model mapping
MODEL_MAPPING = {
    "T5-base weight 1-1": {
        "checkpoint_path": "models/model_2024-03-22-10/checkpoint-epoch=05-val_loss=0.86.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-03-22-10",
        "comment": "T5-base weight 1-1"
    },
    "T5-base weight 12-1": {
        "checkpoint_path": "models/model_2024-04-09-11/checkpoint-epoch=04-val_loss=0.95.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-04-09-11",
        "comment": "T5-base weight 12-1"
    },
    "T5-base weight 13-1": {
        "checkpoint_path": "models/model_2024-04-09-22/checkpoint-epoch=04-val_loss=0.95.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-04-09-22",
        "comment": "T5-base weight 13-1"
    },
    "T5-base weight 20-1": {
        "checkpoint_path": "models/model_2024-04-08-13/checkpoint-epoch=05-val_loss=1.02.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-04-08-13",
        "comment": "T5-base weight 20-1"
    }
}

# Path to the tokenizer directory
tokenizer_dir = "/data/agirard/Projects/Timetravel/models/Tokenizers"

def load_model_and_tokenizer(model_key):
    """
    Load the model and tokenizer based on the provided model key.
    
    Args:
        model_key (str): The key corresponding to the model in the MODEL_MAPPING.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        model_dir (str): The directory where the model is stored.
    """
    # Retrieve model information from the mapping
    model_info = MODEL_MAPPING[model_key]
    checkpoint_path = model_info["checkpoint_path"]
    model_dir = model_info["model_dir"]

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir, legacy=False)
    
    # Load the model from the checkpoint
    model = FlanT5FineTuner.load_from_checkpoint(
        checkpoint_path,
        model_name=CONFIG["model_name"],
        model_dir=model_dir
    )
    
    return model, tokenizer, model_dir

def generate_attentions(model, tokenizer, model_dir):
    """
    Generate and save attention data for each story in the test set.
    
    Args:
        model: The model used for generating attentions.
        tokenizer: The tokenizer used for encoding inputs.
        model_dir (str): The directory where the model is stored.
    """
    # Define the path to the test data
    data_path = Path(CONFIG["data_dir"]) / 'transformed' / CONFIG["test_file"]
    
    # Load test data
    with open(data_path, 'r') as f:
        stories = [json.loads(line) for line in f]

    # Prepare to collect data for the CSV
    csv_data = []

    # Process each story in the test data
    for story in stories:
        story_id = story["story_id"]
        
        # Construct the input sequence for the model
        input_sequence = (
            f"{story['premise']} "
            f"{story['initial']} "
            f"{story['original_ending']} </s> "
            f"{story['premise']} {story['counterfactual']}"
        )
        
        # Tokenize the input sequence
        tokenized_inputs = tokenizer.encode_plus(
            input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
        )

        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)

        # Generate outputs from the model with attentions
        generated_outputs = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=250,
            return_dict_in_generate=True,
            output_attentions=True
        )
        generated_ids = generated_outputs.sequences

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=generated_ids,
            output_attentions=True
        )

        # Extract attention tensors from the outputs
        encoder_attentions = outputs.encoder_attentions
        decoder_attentions = outputs.decoder_attentions
        cross_attentions = outputs.cross_attentions

        encoder_text = tokenizer.convert_ids_to_tokens(input_ids[0])
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_text_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])

        attention_data_path = Path(model_dir) / 'attentions' / str(story_id)
        attention_data_path.mkdir(parents=True, exist_ok=True)

        # Save attention tensors as NumPy arrays
        for i, encoder_attention in enumerate(encoder_attentions):
            np.save(attention_data_path / f'encoder_attentions_layer_{i}.npy', encoder_attention.detach().cpu().numpy())
        for i, decoder_attention in enumerate(decoder_attentions):
            np.save(attention_data_path / f'decoder_attentions_layer_{i}.npy', decoder_attention.detach().cpu().numpy())
        for i, cross_attention in enumerate(cross_attentions):
            np.save(attention_data_path / f'cross_attentions_layer_{i}.npy', cross_attention.detach().cpu().numpy())

        # Save tokens to JSON
        tokens = {
            'encoder_text': encoder_text,
            'generated_text': generated_text,
            'generated_text_tokens': generated_text_tokens
        }
        with open(attention_data_path / 'tokens.json', 'w') as f:
            json.dump(tokens, f)

        print(f"Saved attention data for story {story_id} in {attention_data_path}")

        # Collect data for CSV
        csv_data.append([
            story_id,
            story['premise'],
            story['initial'],
            story['counterfactual'],
            story['original_ending'],
            story['edited_ending'],
            generated_text
        ])

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(csv_data, columns=[
        "StoryID", "Premise", "Initial", "Counterfactual",
        "Original Ending", "Edited Ending", "Generated Text"
    ])
    csv_filename = f"{Path(CONFIG['test_file']).stem}-attention.csv"
    csv_path = Path(model_dir) / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"Saved stories data to {csv_path}")

if __name__ == "__main__":
    # Determine the device to use (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process each model in the model mapping
    for model_key in MODEL_MAPPING:
        try:
            print(f"Processing model: {model_key}")
            model, tokenizer, model_dir = load_model_and_tokenizer(model_key)
            model = model.to(device)
            generate_attentions(model, tokenizer, model_dir)
            print(f"Generated attentions for model {model_key}")
        except Exception as e:
            print(f"Failed to process model {model_key}: {e}")
