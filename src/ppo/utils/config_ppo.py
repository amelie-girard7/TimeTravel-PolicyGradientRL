import os
from pathlib import Path

# ─── Set root directory for the project ────────────────────────────────────────
# If TIMETRAVEL_ROOT is set in the environment, use it; otherwise default to
# three levels up from this file.
ROOT_DIR = Path(
    os.getenv(
        'TIMETRAVEL_ROOT',
        Path(__file__).resolve().parent.parent.parent
    )
)

# ─── Configuration dictionary ───────────────────────────────────────────────────
CONFIG = {
    # === Paths ===
    "root_dir":     ROOT_DIR,                             # Base project folder
    "data_dir":     ROOT_DIR / "data" / "transformed",    # Preprocessed data
    "models_dir":   ROOT_DIR / "models",                  # Where to save model checkpoints
    "logs_dir":     ROOT_DIR / "logs",                    # Local log files
    "dataset_type": "TimeTravel",                         # Logical name for dataset schema

    # === Data Files ===
    "train_file":  "train_supervised_small.json",   # Training split filename
    "dev_file":    "dev_data.json",                 # Validation split filename
    "test_file":   "test_data.json",                # Test split filename

    # === Model Architecture & Optimization ===
    "model_name":     os.getenv('MODEL_NAME', "google/flan-t5-base"),  # HF model ID
    "batch_size":     int(os.getenv('BATCH_SIZE', 8)),   # Samples per forward/backward pass
    "num_workers":    int(os.getenv('NUM_WORKERS', 4)),  # DataLoader worker threads
    "learning_rate":  float(os.getenv('LEARNING_RATE', 2e-6)),  # LR for the policy (T5)
    "value_lr":       float(os.getenv('VALUE_LR', 1e-6)),      # LR for the value head
    "min_lr":         1e-6,  # Minimum LR floor for scheduler

    # === Tokenization & Generation ===
    "max_length":         512,     # Encoder input max tokens
    "max_gen_length":     250,     # Decoder max new tokens at generation time
    "temperature":        0.7,     # Sampling “temperature”
    "top_k":              50,      # Top‑K sampling filter
    "top_p":              0.90,     # Nucleus (top‑p) sampling filter


    # === PPO Core Parameters ===
    "ppo_experiment":        "delta_m1",     # Reward‑shaping variant
    "training_epochs":       int(os.getenv("TRAINING_EPOCHS", 8)),  # Epochs over entire dataset
    "ppo_epochs":            4,       # Inner‑loop gradient passes per PPO update
    "ppo_clip_epsilon":      0.1,     # Clip range ε for policy ratio
    "entropy_coef":          0.01,    # Weight on entropy bonus
    "value_coef":            0.3,     # Weight on value loss term
    "max_trajectory_length": 1024,     # Token budget before triggering PPO update
    "gamma":                 0.99,    # Discount factor for rewards , Higher discount for longer sequences
    "lambda":                0.95,    # GAE mixing parameter

    # === Reward Calculation ===
    "reward_metric":   "bart",  # Which text metric to use for reward


    # === Precision & Gradient Clipping ===
    "precision":          "32-true",  # Training precision: "32-true" or "16-mixed"
    "gradient_clip_val":  0.5,        # Clip gradients by global norm

    # === Training Optimization ===
    "accumulate_grad_batches": 1,   # Gradient accumulation steps
    "overfit_batches": 0,   # Set >0 for debugging on few batches

    # === Initialization & Checkpoints ===
    "ppo_from_checkpoint": True,        # Load existing PPO checkpoint?
    "ppo_checkpoint_path": "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2025-04-17-22/checkpoint-epoch=01-val_loss=0.80.ckpt",
    "ppo_resume_training": False,       # Resume optimizer/scheduler state

    # === Evaluation Metrics ===
    # "use_bert":               True,  # Compute BERT‑based score at eval
    # "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",

    # "scorer_device":          "cuda:0",
    # "bert_scorer_batch_size": 1,
    # "use_bleu":               True,  # Compute BLEU at eval


    "use_bart":               True,  # Compute BART‑based summary score at eval
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",

    # === Debugging ===
    "log_samples_every_n_steps":   100,    # How often to log decoded samples

    # === To remove after checking ======
    "repetition_penalty": 1.2,     # Penalty to reduce repeats
    "shuffle":            True,    # Shuffle train set each epoch
    "results_dir":  ROOT_DIR / "results",                 # Evaluation outputs (CSVs, plots)
    "reward_margin":         0.1,     # Contrastive reward margin
    # "reward_clip_min": -5.0,    # Lower bound clip on raw reward
    # "reward_clip_max":  5.0,    # Upper bound clip on raw reward
    "max_grad_norm":    0.5,    # (Legacy) max gradient norm for clipping
    "cache_teacher_states":     True,  # Cache MLE states for faster reward calcs
    "init_value_head":     "from_mle",  # "random" or "from_mle"
    "output_attentions":           False,  # Return attention maps if True
}

# ─── Ensure required directories exist ─────────────────────────────────────────
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
