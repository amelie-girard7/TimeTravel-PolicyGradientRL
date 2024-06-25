from flask import Flask, jsonify, request, render_template, send_file, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from bertviz import model_view, head_view

app = Flask(__name__)

# Path to the CSV data file
DATA_PATH = Path('/data/agirard/Projects/Timetravel/models/model_2024-03-22-10/test_data_sample-attention.csv')

# Path to the attention data folder
ATTENTION_PATH = Path('/data/agirard/Projects/Timetravel/models/model_2024-03-22-10/attentions')

# Function to load data from the CSV file
def load_data():
    if DATA_PATH is None or not DATA_PATH.exists():
        return None
    print(f"Loading data from {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

# Function to get attention data
def get_attention_data(attention_path, story_id):
    attention_dir = attention_path / str(story_id)
    print(f"Loading attention data from {attention_dir}")

    encoder_attentions = [np.load(attention_dir / f'encoder_attentions_layer_{i}.npy') for i in range(12)]
    decoder_attentions = [np.load(attention_dir / f'decoder_attentions_layer_{i}.npy') for i in range(12)]
    cross_attentions = [np.load(attention_dir / f'cross_attentions_layer_{i}.npy') for i in range(12)]
    
    with open(attention_dir / "tokens.json") as f:
        tokens = json.load(f)
    
    encoder_text = tokens['encoder_text']
    generated_text = tokens['generated_text']
    generated_text_tokens = tokens['generated_text_tokens']
    
    return encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens

# Function to normalize attention weights
def normalize_attention(attention):
    if isinstance(attention, tuple):
        print("Attention is a tuple, extracting the first element.")
        attention = attention[0]

    print("Initial attention shape:", attention.shape)
    print("Averaging attention weights across all heads.")
    normalized_attention = attention.mean(axis=1)
    print("Normalized attention shape:", normalized_attention.shape)
    
    return normalized_attention

# Function to plot attention heatmap
def plot_attention_heatmap(attention, x_tokens, y_tokens, title, image_path):
    # Ensure that the attention matrix matches the dimensions of the token lists
    print(f"Number of x_tokens (input): {len(x_tokens)}")
    print(f"Number of y_tokens (generated text): {len(y_tokens)}")
    print(f"Attention matrix shape: {attention.shape}")
    
    if attention.shape[-1] != len(x_tokens) or attention.shape[-2] != len(y_tokens):
        print("Attention dimensions do not match the token list dimensions.")
        return
    
    fig_width = max(15, len(x_tokens) / 2)
    fig_height = max(10, len(y_tokens) / 2)
    
    plt.figure(figsize=(fig_width, fig_height))
    print("Attention matrix shape for plotting:", attention.shape)
    print("Number of input tokens:", len(x_tokens))
    print("Number of output tokens:", len(y_tokens))
    
    sns.heatmap(attention, xticklabels=x_tokens, yticklabels=y_tokens, cmap='viridis', cbar=True)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Input Tokens', fontsize=12)
    plt.ylabel('Generated Text Tokens', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    plt.savefig(image_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    models = [{"key": "T5-base weight 1-1", "comment": "T5-base weight 1-1"}]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    data = load_data()
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/fetch_story_data', methods=['POST'])
def fetch_story_data():
    story_index = request.json.get('story_index')
    if story_index is None:
        return jsonify({"error": "Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data = load_data()
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    
    story = data.iloc[story_index].to_dict()
    return jsonify(story)

@app.route('/visualize_attention', methods=['POST'])
def visualize_attention():
    story_index = request.json.get('story_index')
    if story_index is None:
        return jsonify({"error": "Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data = load_data()
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    
    story_id = data.iloc[story_index]["StoryID"]
    
    try:
        encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens = get_attention_data(ATTENTION_PATH, story_id)
        print(f"Attention data loaded for story index {story_index}")
        print(f"Generated Text Tokens: {generated_text_tokens}")
    except Exception as e:
        print(f"Error loading attention data: {str(e)}")
        return jsonify({"error": str(e)}), 500

    try:
        first_layer_attention = normalize_attention(cross_attentions[0])
        first_batch_attention = first_layer_attention[0]
        print("Shape of first batch attention:", first_batch_attention.shape)
        
        if first_batch_attention.ndim == 3:
            attention_to_plot = first_batch_attention.mean(axis=0)
            print("Averaged attention shape:", attention_to_plot.shape)
        elif first_batch_attention.ndim == 2:
            attention_to_plot = first_batch_attention
        else:
            print(f"Unexpected attention matrix dimension: {first_batch_attention.ndim}D")
            raise ValueError(f"Unexpected attention matrix dimension: {first_batch_attention.ndim}D")

        image_path = f'/tmp/attention_heatmap_{story_id}.png'
        plot_attention_heatmap(attention_to_plot, encoder_text, generated_text_tokens, "Cross-Attention Weights (First Layer)", image_path)
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"image_path": image_path})

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('/tmp', filename)

if __name__ == '__main__':
    app.run(debug=True)
