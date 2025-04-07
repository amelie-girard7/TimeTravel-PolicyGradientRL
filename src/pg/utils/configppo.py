# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/config_ppo.py
import os
from pathlib import Path

# Root directory for the project (override with TIMETRAVEL_ROOT if set)
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

CONFIG = {
    # === Paths ===
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "results_dir": ROOT_DIR / "results",
    "dataset_type": "TimeTravel",  # Dataset format: "TimeTravel", "ART", etc.

    # === Data Files ===
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

    # === Model ===
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "batch_size": int(os.getenv('BATCH_SIZE', 16)),           # Batch size (PPO needs larger batch for stability)
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),          # Number of data loader workers
    "learning_rate": float(os.getenv('LEARNING_RATE', 1e-5)), # Learning rate for policy/value updates

    # === Tokenization & Generation ===
    "max_length": 512,                # Max input sequence length
    "shuffle": True,                  # Shuffle data during training
    "max_gen_length": 250,           # Max generation length for the model
    "temperature": 0.7,              # Sampling temperature during generation

    # === PPO-Specific Parameters ===
    "ppo_experiment": "delta_m1",    # Reward shaping method: "delta_m1", "SCST", etc.
    "ppo_epochs": 4,                 # PPO epochs per trajectory batch
    "ppo_clip_epsilon": 0.2,         # Clipping epsilon for PPO ratio
    "entropy_coef": 0.01,            # Coefficient for entropy bonus (exploration)
    "value_coef": 0.5,               # Coefficient for value function loss
    "max_trajectory_length": 1024,   # Number of samples before PPO update
    "gamma": 0.99,                   # Discount factor for future rewards
    "lambda": 0.95,                  # GAE lambda (bias/variance trade-off)

    # === PPO Initialization Mode ===
    "init_value_head": "from_mle",   # Init value head: "random" or "from_mle"

    # === Reward Signal Metric ===
    "reward_metric": "bart",         # Options: "bart", "rouge", "bert", "bleu"

    # === Optional: Objective Clipping ===
    "objective_clipping": False,     # Clip rewards to [0, ∞) if True

    # === Resume PPO from a Pretrained Checkpoint (usually from MLE) ===
    "ppo_from_checkpoint": True,     # If True, load weights from MLE checkpoint
    "ppo_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-02-13/checkpoint-epoch=04-val_loss=2.04.ckpt",
    #"ppo_checkpoint_path": None,

    # === Metric Scorers ===
    "use_bert": True,
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 1,

    "use_bleu": True,
    "use_bart": True,
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",

    # === Optional Debug ===
    "output_attentions": False,      # Enable to inspect attention maps (for debugging)
}

# Create output directories if not already existing
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
