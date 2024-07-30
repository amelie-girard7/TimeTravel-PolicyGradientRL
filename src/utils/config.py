#src/utils/config.py

import os
from pathlib import Path

# Allow the root directory to be set via an environment variable for flexibility
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))
BARTSCORE_DIR = ROOT_DIR / "src" / "BARTScore_metric"

# Configuration parameters
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR, 
    "data_dir": ROOT_DIR / "data",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "bart_score_dir": BARTSCORE_DIR,
    "results_dir": ROOT_DIR / "results",  # Directory to save GPT inference results
    
    # File names
    #"train_file": "train_supervised_small.json",
    #"dev_file": "dev_data.json",
    #"test_file": "gold_data.json",

    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",
    
    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    #"model_name": os.getenv('MODEL_NAME', "google/flan-t5-large"),
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "max_epochs": int(os.getenv('MAX_EPOCHS', 6)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),
    "use_custom_loss": True,  # True if you want to use custom loss function
    "output_attentions": True,  # Enable/disable attention outputs
    "log_attentions": True, # True if you want to log the attention
    
    # preprocess data parameters
    "max_length": 512,

    # Text generation parameters
    "max_gen_length": 250,
   

    # Evaluation metrics settings
    "eval_batch_size": 4,
    
    # BERTScorer settings
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 4,

    # BARTScorer settings
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",

    # GPT Inference and evaluation settings
    "inference_mode": "zero_shot",  # Options: zero_shot, one_shot
    "run_similarities_only": False,  # If True, only run similarities
    "example_selection": "random"  # or "random" - Example selection for one_shot mode

}

# Optionally, validate or create the directories
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True) 