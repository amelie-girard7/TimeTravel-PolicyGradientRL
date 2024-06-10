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
from IPython.display import display, HTML

app = Flask(__name__)

# Set the specific GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust the index to select a different GPU

# Check if CUDA is available and set the device
if torch.cuda.is_available():
    print("CUDA is available. Configuring to use GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# Setup tokenizer directory to use the existing tokenizer
tokenizer_dir = "/data/agirard/Projects/Timetravel/models/Tokenizers"

# Load the tokenizer from the specified directory
print("Loading tokenizer from the local directory...")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir, legacy=False)
print("Tokenizer loaded.")

# Path to the checkpoint file
checkpoint_path = "/data/agirard/Projects/Timetravel/models/model_2024-05-14-20/checkpoint-epoch=00-val_loss=8.20.ckpt"

# Load the model from the checkpoint
print("Loading model from the checkpoint...")
model = FlanT5FineTuner.load_from_checkpoint(
    checkpoint_path,
    model_name=CONFIG["model_name"],
    model_dir="/data/agirard/Projects/Timetravel/models/model_2024-05-14-20"
)
model = model.to(device)  # Move model to the correct device
print("Model loaded and moved to device.")

# Function to setup dataloaders
def setup_dataloaders(tokenizer):
    data_path = CONFIG["data_dir"] / 'transformed'
    batch_size = CONFIG["batch_size"]
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
dev_key = CONFIG["dev_file"].split('.')[0]  # 'dev_data_sample'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stories', methods=['GET'])
def get_stories():
    data_path = Path(CONFIG["data_dir"]) / 'transformed' / CONFIG["dev_file"]
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
    data_path = Path(CONFIG["data_dir"]) / 'transformed' / CONFIG["dev_file"]
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
    tokenized_ending = tokenizer.encode_plus(
        story['edited_ending'], truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
    )

    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)
    labels = tokenized_ending['input_ids'].to(device)

    # Generate text and capture attention weights using the forward function
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # Extract attention tensors
    encoder_attentions = outputs.encoder_attentions
    decoder_attentions = outputs.decoder_attentions
    cross_attentions = outputs.cross_attentions

    # Convert Input IDs to Tokens
    encoder_text = tokenizer.convert_ids_to_tokens(input_ids[0])
    decoder_text = tokenizer.convert_ids_to_tokens(labels[0])

    return encoder_attentions, decoder_attentions, cross_attentions, encoder_text, decoder_text

@app.route('/visualize_attention', methods=['POST'])
def visualize_attention():
    story_id = request.json.get('story_id')
    data = fetch_story_data(story_id)
    if data is None:
        return jsonify({"error": "Story not found"}), 404
    
    encoder_attentions, decoder_attentions, cross_attentions, encoder_text, decoder_text = data

    # Normalize attention weights
    def normalize_attention(attention):
        if isinstance(attention, tuple):
            attention = attention[0]
        normalized_attention = attention.mean(dim=1).detach().cpu().numpy()
        return normalized_attention

    last_layer_attention = normalize_attention(cross_attentions[-1])
    attention_to_plot = last_layer_attention[0]

    # Plot the attention heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(attention_to_plot, xticklabels=encoder_text, yticklabels=decoder_text, cmap='viridis', cbar=True)
    plt.xticks(rotation=90)
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title('Cross-Attention Weights (Last Layer)')
    
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
    encoder_attentions, decoder_attentions, cross_attentions, encoder_text, decoder_text = data

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
        decoder_tokens=decoder_text,
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

    encoder_attentions, decoder_attentions, cross_attentions, encoder_text, decoder_text = data

    if encoder_attentions is None or decoder_attentions is None or cross_attentions is None:
        return jsonify({"error": "Attentions were not captured correctly."}), 500

    # Capture HTML content from head_view
    html_content = head_view(
        encoder_attention=encoder_attentions,
        decoder_attention=decoder_attentions,
        cross_attention=cross_attentions,
        encoder_tokens=encoder_text,
        decoder_tokens=decoder_text,
        layer=11,
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
