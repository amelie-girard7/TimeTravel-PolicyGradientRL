import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attention, x_tokens, y_tokens, title, image_path):
    """
    Plot and save an attention heatmap.
    
    Parameters:
    attention (numpy.ndarray): The attention weights to be visualized.
    x_tokens (list of str): The input tokens.
    y_tokens (list of str): The generated text tokens.
    title (str): The title for the heatmap.
    image_path (str): The path to save the generated heatmap image.
    """
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
