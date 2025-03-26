# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/utils/config.py
import os
from pathlib import Path

# Set the root directory based on an environment variable or default to a parent directory
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

# Configuration dictionary for model training, paths, and other settings
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",  # Directory containing transformed data
    "models_dir": ROOT_DIR / "models",  # Directory to save models
    "logs_dir": ROOT_DIR / "logs",  # Directory for logs
    "results_dir": ROOT_DIR / "results",  # Directory for results (e.g., validation details)
    "dataset_type": "TimeTravel",  # Options: "ART", "TimeTravel", "AblatedTimeTravel"

    # ******** Data files***********
    # Sample Timetravel sample datasets
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

    # Timetravel,AblatedTimeTravel datasets
    #"train_file": "train_supervised_small.json",
    #"dev_file": "dev_data.json",
    #"test_file": "test_data.json",

    # Sample Art dataset
    #"train_file": "art_train_data_sample.json",
    #"dev_file": "art_dev_data_sample.json",
    #"test_file": "art_test_data_sample.json", 
    # 
    # Art dataset
    #"train_file": "art_train_data.json",
    #"dev_file": "art_dev_data.json",
    #"test_file": "art_test_data.json",    

    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),  # Hugging Face model to load
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Number of samples per batch
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),  # Number of workers for data loading
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),  # Learning rate for the optimizer

    # Preprocessing and generation parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle the data during training
    "max_gen_length": 250,  # Maximum length for generated text

    # Additional training options
    "use_custom_loss": False,  # Whether to use a custom loss function (set to False for MLE)
    "output_attentions": False,  # Set to True to output attentions from the model (optional)
    # Additional configuration for scoring metrics
    "reward_metric": "bert",  # "rouge","bart", "bert","bleu" (default to "rouge")
    # Add temperature for sampling
    "temperature": 0.7,  # Temperature for sampling (default: 0.7)

    # **Experiment Selection**
    "pg_experiment": "dynamic",  # Options: "fixed", "dynamic", "delta_m1"
    "delta_m1_enabled": False,  # Enable Delta_M1 reward adjustments
    "baseline_score": 0.5,  # Used for PG fixed baseline experiment

    # PG Training
    # "pg_from_checkpoint": True,
    # "pg_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/mle_2025-03-18-12/checkpoint-epoch=00-val_loss=1.00.ckpt",   # MLE1_TT
    # "pg_epochs": 5,  # Number of epochs to fine-tune with PG

    # "pg_from_checkpoint": True,
    # "pg_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/mle_2025-03-18-13/checkpoint-epoch=01-val_loss=0.98.ckpt",   # MLE2_TT
    # "pg_epochs": 4,  # Number of epochs to fine-tune with PG

    "pg_from_checkpoint": True,
    "pg_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2024-03-22-10/checkpoint-epoch=05-val_loss=0.86.ckpt",   # MLE6_TT
    "pg_epochs": 1,  # Number of epochs to fine-tune with PG

    # New configuration for PG objective modifications:
    "objective_clipping": False,  # Disable reward clipping (keep raw rewards as computed)
    "use_greedy_reward": True,   # Disable greedy decoding (use the sampled outputs as before)
  
    # Additional configuration for scoring metrics 
    "use_bert": True,  # Disable BERT scorer
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",  # Default BERT model for scorer 
    "scorer_device": "cuda:0",  # Device for the scorer
    "bert_scorer_batch_size": 4,  # Batch size for BERT scorer 

    "use_bleu": True,  # Disable BLEU scorer,

    "use_bart": True,  # Disable BART scorer
    "bart_scorer_checkpoint": "facebook/bart-large-cnn"  # Default BART model for scorer
}

# Create any directories that don't exist
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
