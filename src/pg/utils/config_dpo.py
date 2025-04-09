import os
from pathlib import Path
from typing import Dict, Any

# Root directory for the project (override with TIMETRAVEL_ROOT if set)
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

CONFIG: Dict[str, Any] = {
    # === System Configuration ===
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "results_dir": ROOT_DIR / "results",
    "seed": 42,
    "deterministic": True,
    "precision": "16-mixed",  # "32-true", "16-mixed", "bf16-mixed"
    
    # === Data Configuration ===
    "dataset_type": "TimeTravel",  # "TimeTravel", "ART", "AblatedTimeTravel"
    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",
    "max_length": 512,  # Maximum input sequence length
    "max_gen_length": 128,  # Maximum generation length
    
    # === Model Configuration ===
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "tokenizer_config": {
        "use_fast": True,
        "legacy": False
    },
    
    # === Training Configuration ===
    "epochs": 10,
    "batch_size": int(os.getenv('BATCH_SIZE', 8)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 1e-5)),
    "weight_decay": 0.01,
    "gradient_clip_val": 0.5,
    "gradient_clip_algorithm": "norm",
    "val_check_interval": 0.25,  # Validate 4 times per epoch
    "accumulate_grad_batches": 1,
    
    # === DPO-Specific Configuration ===
    "dpo": {
        "beta": 0.1,  # KL divergence coefficient
        "loss_type": "sigmoid",  # "sigmoid" or "hinge"
        "beta_schedule": {
            "initial": 0.1,
            "final": 0.5,
            "num_steps": 1000
        },
        "reference_model": {
            "freeze": True,
            "sync_with_policy": False
        }
    },
    
    # === Generation Configuration ===
    "generation": {
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.8,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1.0,
        "num_beams": 1,
        "do_sample": True,
        "early_stopping": True,
        "bad_words_ids": None  # Can block original ending tokens
    },
    
    # === Metrics Configuration ===
    "metrics": {
        "primary": "bart",  # Primary metric for evaluation
        "secondary": ["rouge", "bleu"],
        "device": "cuda:0",
        "batch_size": 4
    },
    
    # === Scorers Configuration ===
    "scorers": {
        "bert": {
            "enable": True,
            "model_type": "microsoft/deberta-xlarge-mnli",
            "batch_size": 1
        },
        "bart": {
            "enable": True,
            "checkpoint": "facebook/bart-large-cnn"
        },
        "bleu": {
            "enable": True
        },
        "rouge": {
            "enable": True
        }
    },
    
    # === Logging & Monitoring ===
    "monitoring": {
        "log_every_n_steps": 10,
        "track_grad_norm": -1,  # -1 to disable, 2 for L2 norm
        "progress_bar_refresh_rate": 1
    },
    
    # === Debugging ===
    "debug": {
        "output_attentions": False,
        "overfit_batches": 0,  # 0 to disable, >0 to overfit
        "fast_dev_run": False
    }
}

def initialize_directories(config: Dict[str, Any]) -> None:
    """Ensure all required directories exist"""
    for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
        path = config[path_key]
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)

# Initialize directories
initialize_directories(CONFIG)

# Environment variable overrides
if "BATCH_SIZE" in os.environ:
    CONFIG["batch_size"] = int(os.environ["BATCH_SIZE"])
if "LEARNING_RATE" in os.environ:
    CONFIG["learning_rate"] = float(os.environ["LEARNING_RATE"])
if "MODEL_NAME" in os.environ:
    CONFIG["model_name"] = os.environ["MODEL_NAME"]