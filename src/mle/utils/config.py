# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/mle/utils/config.py
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
    "results_dir": ROOT_DIR / "results",  # Directory to save the results
    "dataset_type": "TimeTravel",  # Options: "ART", "TimeTravel", "AblatedTimeTravel"

    # Timetravel,AblatedTimeTravel datasets
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",
    
    #"test_file": "gold_data.json",


    # Art dataset
    # "train_file": "art_train_data.json",
    # "dev_file": "art_dev_data.json",
    # "test_file": "art_test_data.json",  
    
    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    #"model_name": os.getenv('MODEL_NAME', "google/flan-t5-large"),
    "batch_size": int(os.getenv('BATCH_SIZE', 8)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "max_epochs": int(os.getenv('MAX_EPOCHS', 10)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),
    "use_custom_loss": False,  # True if you want to use custom loss function
    "output_attentions": False,  # Enable/disable attention outputs
    "log_attentions": False, # True if you want to log the attention
    
    # preprocess data parameters
    "max_length": 512,

    # Text generation parameters
    "max_gen_length": 250,

    "temperature": 0.7,  # 0.1-1.0, higher = more random
    "top_k": 50,        # Consider top 50 tokens
    "top_p": 0.9,       # Nucleus sampling threshold


    # Evaluation metrics settings
    "eval_batch_size": 1,
    
    # BERTScorer settings
    "use_bert": True,  # Add this to control BERT usage
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 4,

    # BARTScorer settings
    "use_bart": True,  # Add this to control BART usage
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",

    # GPT Inference and evaluation settings
    # "inference_mode": "zero_shot",  # Options: zero_shot, one_shot
    # "example_selection": "fixed",  # "fixed" or "random" - Example selection for one_shot mode
    # "run_similarities_only": True  # If True, only run similarities, # False, Generate new results

    # Add these new parameters
    "value_head_hidden_size": 512,
    "save_full_checkpoint": True  # Ensure we save all model parts



}

# Optionally, validate or create the directories
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
