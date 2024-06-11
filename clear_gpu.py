import torch

def clear_gpu_cache():
    """Function to clear the GPU cache."""
    if torch.cuda.is_available():
        print("CUDA is available. Clearing GPU cache.")
        torch.cuda.empty_cache()
        print("GPU cache cleared.")

clear_gpu_cache()
