from flask import Flask, jsonify, request, render_template, send_file
import os
import torch
from transformers import T5Tokenizer
from src.models.model_T5 import FlanT5FineTuner
from src.utils.config import CONFIG
from src.data_loader import create_dataloaders
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bertviz import model_view, head_view

app = Flask(__name__)

# Set the specific GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Adjust the index to select a different GPU

# Check if CUDA is available and set the device
if torch.cuda.is_available():
    print("CUDA is available. Configuring to use GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Function to clear GPU cache
def clear_gpu_cache():
    if torch.cuda.is_available():
        print("Clearing GPU cache...")
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

# Setup tokenizer directory to use the existing tokenizer
tokenizer_dir = "/data/agirard/Projects/Timetravel/models/Tokenizers"

# Load the tokenizer from the specified directory
print("Loading tokenizer from the local directory...")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir, legacy=False)
print("Tokenizer loaded.")

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
    },
    "T5-large weight 1-1": {
        "checkpoint_path": "models/model_2024-03-22-15/checkpoint-epoch=02-val_loss=0.78.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-03-22-15",
        "comment": "T5-large weight 1-1"
    },
    "T5-large weight 15-1": {
        "checkpoint_path": "models/model_2024-04-10-10/checkpoint-epoch=03-val_loss=0.89.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-04-10-10",
        "comment": "T5-large weight 15-1"
    },
    "T5-large weight 20-1": {
        "checkpoint_path": "models/model_2024-04-08-09/checkpoint-epoch=03-val_loss=0.91.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-04-08-09",
        "comment": "T5-large weight 20-1"
    },
    "T5-large weight 30-1": {
        "checkpoint_path": "models/model_2024-04-10-14/checkpoint-epoch=04-val_loss=0.98.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-04-10-14",
        "comment": "T5-large weight 30-1"
    },
    "T5-large (Gold data) weight 20-1": {
        "checkpoint_path": "models/model_2024-05-14-20/checkpoint-epoch=00-val_loss=8.20.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-05-14-20",
        "comment": "T5-large (Gold data) weight 20-1"
    },
    "T5-base (Gold data) weight 13-1": {
        "checkpoint_path": "models/model_2024-05-13-17/checkpoint-epoch=05-val_loss=0.98.ckpt",
        "model_dir": "/data/agirard/Projects/Timetravel/models/model_2024-05-13-17",
        "comment": "T5-base (Gold data) weight 13-1"
    }
}

# Function to load model and move it to the correct device
def load_model(model_key):
    if model_key not in MODEL_MAPPING:
        raise ValueError(f"Model {model_key} not found in the mapping.")
    model_info = MODEL_MAPPING[model_key]
    checkpoint_path = model_info["checkpoint_path"]
    model_dir = model_info["model_dir"]
    comment = model_info["comment"]
    
    print(f"Loading model: {comment} from the checkpoint {checkpoint_path}")
    clear_gpu_cache()  # Clear the cache to free up memory
    model = FlanT5FineTuner.load_from_checkpoint(
        checkpoint_path,
        model_name=CONFIG["model_name"],
        model_dir=model_dir
    )
    model = model.to(device)  # Move model to the correct device
    clear_gpu_cache()  # Clear the cache again after loading
    print("Model loaded and moved to device.")
    return model, comment

# Attempt to load the initial model and handle out-of-memory errors
initial_model_key = "T5-large (Gold data) weight 20-1"
try:
    model, model_comment = load_model(initial_model_key)
except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory. Attempting to free up space and retry.")
    clear_gpu_cache()  # Clear the cache
    model, model_comment = load_model(initial_model_key)

# Move BERTScorer model to the correct device
model.bert_scorer._model.to(device)

# Function to setup dataloaders
def setup_dataloaders(tokenizer):
    data_path = CONFIG["data_dir"] / 'transformed'
    batch_size = CONFIG.get("batch_size", 1) // 2  # Ensure batch_size is at least 1
    batch_size = max(batch_size, 1)  # Ensure batch_size is a positive integer
    num_workers = CONFIG["num_workers"]
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(data_path, tokenizer, batch_size, num_workers)
    print("Dataloaders created.")
    return dataloaders

# Prepare dataloaders
print("Setting up dataloaders...")
dataloaders = setup_dataloaders(tokenizer)
print("Dataloaders setup completed.")

# Extract the keys for train, dev, and test from CONFIG and remove the file extension
test_key = CONFIG["test_file"].split('.')[0]  # 'dev_data_sample'

# Updated index route to include the model comment
@app.route('/')
def index():
    return render_template('index.html', model_comment=model_comment)

@app.route('/get_models', methods=['GET'])
def get_models():
    models = [{"key": key, "comment": value["comment"]} for key, value in MODEL_MAPPING.items()]
    return jsonify(models)

@app.route('/load_model', methods=['POST'])
def load_selected_model():
    model_key = request.json.get('model_key')
    try:
        global model, model_comment
        model, model_comment = load_model(model_key)
        return jsonify({"status": "success", "comment": model_comment})
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/get_stories', methods=['GET'])
def get_stories():
    data_path = Path(CONFIG["data_dir"]) / 'transformed' / CONFIG["test_file"]
    stories = []
    try:
        with open(data_path, 'r') as f:
            for line in f:
                stories.append(json.loads(line))
        return jsonify(stories)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return jsonify({"error": "Stories file not found"}), 404

def fetch_story_data(story_id):
    data_path = Path(CONFIG["data_dir"]) / 'transformed' / CONFIG["test_file"]
    with open(data_path, 'r') as f:
        stories = [json.loads(line) for line in f]
    
    story = next((s for s in stories if s["story_id"] == story_id), None)
    if not story:
        return None
    
    # Preprocess the story
    input_sequence = (
        f"{story['premise']}"
        f"{story['initial']}"
        f"{story['original_ending']} </s> "
        f"{story['premise']} {story['counterfactual']}"
    )
    tokenized_inputs = tokenizer.encode_plus(
        input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
    )

    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)

    # Generate text using the model's generate method
    print("Generating text using the model's generate method...")
    generated_outputs = model.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=250,
        return_dict_in_generate=True,
        output_attentions=True
    )
    generated_ids = generated_outputs.sequences  # Extract generated sequences
    print("Text generation completed.")

    # Extracting attentions using the underlying model's forward method
    print("Extracting attentions using the underlying model's forward method...")
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=generated_ids,
        output_attentions=True
    )
    print("Attentions extracted.")

    # Extract attention tensors from the output
    encoder_attentions = outputs.encoder_attentions
    decoder_attentions = outputs.decoder_attentions
    cross_attentions = outputs.cross_attentions

    # Convert Input IDs to Tokens
    encoder_text = tokenizer.convert_ids_to_tokens(input_ids[0])
    generated_text = tokenizer.convert_ids_to_tokens(generated_ids[0])  # Changed to generated_ids

    return encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, story

@app.route('/visualize_attention', methods=['POST'])
def visualize_attention():
    story_id = request.json.get('story_id')
    data = fetch_story_data(story_id)
    if data is None:
        return jsonify({"error": "Story not found"}), 404
    
    encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, _ = data

    # Normalize attention weights
    def normalize_attention(attention):
        if isinstance(attention, tuple):
            attention = attention[0]
        normalized_attention = attention.mean(dim=1).detach().cpu().numpy()
        return normalized_attention

    first_layer_attention = normalize_attention(cross_attentions[0])  # Use the first layer's cross-attention
    attention_to_plot = first_layer_attention[0]

    # Plot the attention heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(attention_to_plot, xticklabels=encoder_text, yticklabels=generated_text, cmap='viridis', cbar=True)
    plt.xticks(rotation=90)
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title('Cross-Attention Weights (First Layer)')
    
    # Save the plot as an image
    image_path = '/tmp/attention_heatmap.png'
    plt.savefig(image_path)
    plt.close()

    return send_file(image_path, mimetype='image/png')

@app.route('/visualize_model_view', methods=['POST'])
def visualize_model_view():
    print("Received request for visualize_model_view")
    
    # Get the story ID from the request
    story_id = request.json.get('story_id')
    print(f"Story ID: {story_id}")

    # Fetch story data
    data = fetch_story_data(story_id)
    if data is None:
        print("Story not found")
        return jsonify({"error": "Story not found"}), 404

    # Unpack the data
    encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, _ = data

    # Check if attentions were captured
    if encoder_attentions is None or decoder_attentions is None or cross_attentions is None:
        print("Attentions were not captured correctly.")
        return jsonify({"error": "Attentions were not captured correctly."}), 500

    # Print the types and shapes of the attention tensors for debugging
    print(f"Type of encoder_attentions: {type(encoder_attentions)}")
    print(f"Type of decoder_attentions: {type(decoder_attentions)}")
    print(f"Type of cross_attentions: {type(cross_attentions)}")
    print(f"Shape of encoder_attentions: {[a.shape for a in encoder_attentions]}")
    print(f"Shape of decoder_attentions: {[a.shape for a in decoder_attentions]}")
    print(f"Shape of cross_attentions: {[a.shape for a in cross_attentions]}")

    # Capture HTML content from model_view
    print("Capturing HTML content from model_view...")
    html_content = model_view(
        encoder_attention=encoder_attentions,
        decoder_attention=decoder_attentions,
        cross_attention=cross_attentions,
        encoder_tokens=encoder_text,
        decoder_tokens=generated_text,
        html_action='return'
    ).data
    print("HTML content captured")

    # Check the type and length of html_content
    print(f"Type of html_content: {type(html_content)}")
    print(f"Length of html_content: {len(html_content)}")

    # Write HTML content to file
    html_path = '/tmp/model_view.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML content written to {html_path}")

    return send_file(html_path, mimetype='text/html')

@app.route('/visualize_head_view', methods=['POST'])
def visualize_head_view():
    story_id = request.json.get('story_id')
    data = fetch_story_data(story_id)
    if data is None:
        return jsonify({"error": "Story not found"}), 404

    encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, _ = data

    if encoder_attentions is None or decoder_attentions is None or cross_attentions is None:
        return jsonify({"error": "Attentions were not captured correctly."}), 500

    # Capture HTML content from head_view
    html_content = head_view(
        encoder_attention=encoder_attentions,
        decoder_attention=decoder_attentions,
        cross_attention=cross_attentions,
        encoder_tokens=encoder_text,
        decoder_tokens=generated_text,
        layer=0,  # Specify the first layer
        html_action='return'
    ).data

    # Write HTML content to file
    html_path = '/tmp/head_view.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return send_file(html_path, mimetype='text/html')

@app.route('/model_view')
def serve_model_view():
    return send_file('/tmp/model_view.html', mimetype='text/html')

@app.route('/head_view')
def serve_head_view():
    return send_file('/tmp/head_view.html', mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
