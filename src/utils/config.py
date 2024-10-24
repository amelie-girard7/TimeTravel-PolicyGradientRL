# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/utils/config.py
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
    
    # File names
    #"train_file": "train_supervised_small_sample.json",
    #"dev_file": "dev_data_sample.json",
    #"test_file": "test_data_sample.json",

    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",
    
    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2024-03-22-10/checkpoint-epoch=05-val_loss=0.86.ckpt",  # Updated checkpoint path
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "max_epochs": int(os.getenv('MAX_EPOCHS', 3)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),
    

    # Preprocess data parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle data during training

    # Text generation parameters
    "max_gen_length": 250,  # Maximum length for generated text

    # Reward-based training configuration
    "reward_metric": os.getenv("REWARD_METRIC", "rouge"),  # Can be "rouge", "bleu", "bert", bart.
    "baseline_score": float(os.getenv("BASELINE_SCORE", 0.7)),  # 0.5,0.3,0.7,0.8
   
    # BERTScorer settings
    "use_bert": False,  # Enable BERT usage
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli", 
    "scorer_device": "cuda:0",  # Device for BERT scorer
    "bert_scorer_batch_size": 1,

    # BARTScorer settings
    "use_bart": False,  # Enable BART usage
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",  # BART model checkpoint

    # BLEU Scorer settings
    "use_bleu": False,  # Enable BLEU usage (this flag allows you to toggle BLEU)
   
    #Output attentions for model interpretability
    "output_attentions": False,  # Set this to True if you want the model to return attention weights

    # NEW FLAG: Toggle between MLE loss and Policy Gradient loss
    "use_policy_gradient": True  # Set to True to use Policy Gradient Loss, False for MLE Loss
}

# Optionally, validate or create the directories
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
