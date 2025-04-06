# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/config_ppo.py
import os
from pathlib import Path

ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

CONFIG = {
    # Paths
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "results_dir": ROOT_DIR / "results",
    "dataset_type": "TimeTravel",

    # Data files
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

    # Model
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "batch_size": int(os.getenv('BATCH_SIZE', 16)),  # Larger PPO batches improve stability
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 1e-5)),  # Lower LR typically helps PPO

    # Preprocessing
    "max_length": 512,
    "shuffle": True,
    "max_gen_length": 250,
    "temperature": 0.7,

    # PPO specific parameters clearly set
    "ppo_experiment": "delta_m1",    # Using delta_m1 as reward formulation
    "ppo_epochs": 4,                 # PPO epochs per trajectory batch
    "ppo_clip_epsilon": 0.2,         # Standard PPO clipping
    "entropy_coef": 0.01,            # Exploration coefficient
    "value_coef": 0.5,               # Value function importance
    "max_trajectory_length": 1024,   # How many samples collected per PPO update
    "gamma": 0.99,                   # Discount factor
    "lambda": 0.95,                  # GAE parameter

    # Reward metric clearly set
    "reward_metric": "bart",

    # Objective clipping (optional)
    "objective_clipping": False,

    # PPO checkpoint clearly from MLE checkpoint (epoch 10 MLE)
    "ppo_from_checkpoint": False,
    "ppo_checkpoint_path": None,  # MLE epoch 10 checkpoint

    #"ppo_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-02-13/checkpoint-epoch=04-val_loss=2.04.ckpt",  # MLE epoch 10 checkpoint

    # Scorers (unchanged)
    "use_bert": True,
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 1,
    "use_bleu": True,
    "use_bart": True,
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",

    # Initialization of PPO value head (random/scratch)
    "init_value_head": "random",

    "output_attentions": False,  # Set to True to output attentions from the model (optional)
}

# Create required directories if they don't exist
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
