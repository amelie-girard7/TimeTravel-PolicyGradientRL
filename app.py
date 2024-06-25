from flask import Flask, jsonify, request, render_template, send_file, send_from_directory, make_response
import pandas as pd
import json
import torch
import os
from pathlib import Path
from heatmap import plot_attention_heatmap
from bertviz_view import visualize_head_view, visualize_model_view

app = Flask(__name__)

# Define paths to the data and attention directories
DATA_PATH = Path('/data/agirard/Projects/Timetravel/models/model_2024-03-22-10/test_data_sample-attention.csv')
ATTENTION_PATH = Path('/data/agirard/Projects/Timetravel/models/model_2024-03-22-10/attentions')

def clear_bertviz_cache():
    """
    Clear the cache files for BERTViz to prevent clutter and potential issues with old cache files.
    """
    try:
        bertviz_files = [f for f in os.listdir('/tmp') if f.startswith('attention_heatmap_')]
        for f in bertviz_files:
            os.remove(os.path.join('/tmp', f))
        print("BERTViz cache cleared successfully.")
    except Exception as e:
        print(f"Error clearing BERTViz cache: {str(e)}")

def load_data():
    """
    Load story data from a CSV file.
    Returns a pandas DataFrame if the file exists, otherwise returns None.
    """
    if DATA_PATH is None or not DATA_PATH.exists():
        print("Data path does not exist.")
        return None
    print(f"Loading data from {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def get_attention_data(attention_path, story_id):
    """
    Load attention data for a given story ID.
    This function reads encoder, decoder, and cross-attention tensors and token data.
    Returns a tuple with attention data and tokens if successful, otherwise returns None.
    """
    attention_dir = attention_path / str(story_id)
    print(f"Loading attention data from {attention_dir}")

    if not attention_dir.exists():
        print(f"Attention directory does not exist: {attention_dir}")
        return None

    try:
        encoder_attentions = [torch.load(attention_dir / f'encoder_attentions_layer_{i}.pt') for i in range(12)]
        decoder_attentions = [torch.load(attention_dir / f'decoder_attentions_layer_{i}.pt') for i in range(12)]
        cross_attentions = [torch.load(attention_dir / f'cross_attentions_layer_{i}.pt') for i in range(12)]
    except Exception as e:
        print(f"Error loading attention tensors: {e}")
        return None

    try:
        with open(attention_dir / "tokens.json") as f:
            tokens = json.load(f)
    except Exception as e:
        print(f"Error loading tokens.json: {e}")
        return None

    encoder_text = tokens.get('encoder_text', [])
    generated_text = tokens.get('generated_text', "")
    generated_text_tokens = tokens.get('generated_text_tokens', [])

    print("Loaded encoder_text:", encoder_text)
    print("Loaded generated_text:", generated_text)
    print("Loaded generated_text_tokens:", generated_text_tokens)

    return encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens

@app.route('/')
def index():
    """
    Render the main index page of the application.
    """
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Return a list of models available for selection.
    """
    models = [{"key": "T5-base weight 1-1", "comment": "T5-base weight 1-1"}]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    """
    Return a list of stories from the loaded data.
    """
    data = load_data()
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/fetch_story_data', methods=['POST'])
def fetch_story_data():
    """
    Return detailed information about a specific story given its index.
    """
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
    """
    Generate and return the attention heatmap for a specific story.
    """
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
        result = get_attention_data(ATTENTION_PATH, story_id)
        if result is None:
            return jsonify({"error": "Error loading attention data"}), 500
        encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens = result
        print(f"Attention data loaded for story index {story_index}")
        print(f"Generated Text Tokens: {generated_text_tokens}")
    except Exception as e:
        print(f"Error loading attention data: {str(e)}")
        return jsonify({"error": str(e)}), 500

    try:
        first_layer_attention = cross_attentions[0]
        if isinstance(first_layer_attention, tuple):
            first_layer_attention = first_layer_attention[0]
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
    """
    Serve the generated heatmap image.
    """
    return send_from_directory('/tmp', filename)

@app.route('/visualize_head_view', methods=['POST'])
def handle_visualize_head_view():
    """
    Handle the request to visualize the head view of the attention mechanism.
    """
    return visualize_head_view(request, load_data, get_attention_data, ATTENTION_PATH)

@app.route('/visualize_model_view', methods=['POST'])
def handle_visualize_model_view():
    """
    Handle the request to visualize the model view of the attention mechanism.
    """
    return visualize_model_view(request, load_data, get_attention_data, ATTENTION_PATH)

if __name__ == '__main__':
    clear_bertviz_cache()  # Clear BERTViz cache at the beginning
    app.run(debug=True)
