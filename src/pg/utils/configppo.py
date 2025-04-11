import os
from pathlib import Path

# Set root directory for the project
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

CONFIG = {
    # === Paths ===
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "results_dir": ROOT_DIR / "results",
    "dataset_type": "TimeTravel",

    # === Data Files ===
    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",

    # === Model Architecture ===
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "batch_size": int(os.getenv('BATCH_SIZE', 16)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 1e-5)),
    "value_lr": float(os.getenv('VALUE_LR', 1e-5)),  # Separate LR for value head
    "min_lr": 1e-6,  # Minimum learning rate for scheduler

    # === Tokenization & Generation ===
    "max_length": 512,
    "shuffle": True,
    "max_gen_length": 250,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2,  # Added to reduce repetition

    # === PPO Core Parameters ===
    "ppo_experiment": "contrastive_ratio",  # Options: "contrastive_ratio", "delta_m1", "default"
    "ppo_epochs": 10,
    "ppo_clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_trajectory_length": 1024,
    "gamma": 0.99,
    "lambda": 0.95,
    "reward_margin": 0.2,  # For contrastive reward margin
    "gradient_clip_val": 0.5,

    # === Reward Calculation ===
    "reward_metric": "bart",  # Options: "bart", "rouge", "bert", "bleu"
    "reward_clip_min": -2.0,  # Minimum reward value
    "reward_clip_max": 2.0,   # Maximum reward value

    # === Training Optimization ===
    "accumulate_grad_batches": 1,
    "cache_teacher_states": True,
    "precision": "32-true",  # Options: "32-true", "16-mixed"
    "overfit_batches": 0,  # Set to >0 for debugging

    # === Initialization ===
    "init_value_head": "from_mle",  # Options: "random", "from_mle"
    "ppo_from_checkpoint": True,
    "ppo_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-02-13/checkpoint-epoch=04-val_loss=2.04.ckpt",
    "ppo_resume_training": False,

    # === Evaluation Metrics ===
    "use_bert": True,
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 1,
    "use_bleu": True,
    "use_bart": True,
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",

    # === Debugging ===
    "output_attentions": False,
    "log_samples_every_n_steps": 100,  # Log samples periodically
}

# Create required directories
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)