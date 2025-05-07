# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/art/train_art.py

import sys
import os
import datetime
import logging
import json
from pathlib import Path
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.pg.art.models.model_art import FlanT5FineTuner
from src.pg.art.data_loader_art import create_dataloaders
from src.pg.art.utils.metrics_art import MetricsEvaluator
from src.pg.art.utils.config_art import CONFIG
import re
import wandb

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def convert_paths_to_strings(config_dict):
    """Convert all Path objects in a dictionary to strings for JSON serialization"""
    converted = {}
    for key, value in config_dict.items():
        if isinstance(value, Path):
            converted[key] = str(value)
        elif isinstance(value, dict):
            converted[key] = convert_paths_to_strings(value)
        else:
            converted[key] = value
    return converted

def setup_logging(model_dir):
    """Set up comprehensive logging to both file and console."""
    log_file = model_dir / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_model(model_dir, file_label="", checkpoint_path=None):
    model_dir = Path(model_dir)  # Ensure Path object
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)  # Ensure Path object
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        model = FlanT5FineTuner.load_from_checkpoint(
            str(checkpoint_path),  # str() for PyTorch compatibility
            model_name=CONFIG["model_name"],
            model_dir=str(model_dir),  # str() for PyTorch compatibility
            file_label=file_label,
            strict=False  # Add this line to ignore unexpected keys
        )
    else:
        logger.info(f"Initializing fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(
            CONFIG["model_name"],
            str(model_dir),  # str() for PyTorch compatibility
            file_label=file_label
        )
    return model

def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger, model_dir):
    model_dir = Path(model_dir)  # Ensure Path object
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=0.1,
        default_root_dir=str(model_dir),  # str() for PyTorch compatibility
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer

def evaluate_and_save(model, dataloader, phase, model_dir, file_label):
    """Run evaluation and save results to CSV and WandB."""
    model_dir = Path(model_dir)
    logger.info(f"Starting {phase} evaluation...")
    
    # Clear previous details
    if phase == "test":
        model.epoch_test_details = []
    else:
        model.epoch_validation_details = []
    
    # Use test instead of validate if phase is test
    if phase == "test":
        trainer = Trainer(accelerator='gpu', devices=1, logger=False)
        results = trainer.test(model, dataloader, verbose=False)
    else:
        trainer = Trainer(accelerator='gpu', devices=1, logger=False)
        results = trainer.validate(model, dataloader, verbose=False)
    
    # Log to WandB
    wandb.log({f"{phase}_metrics": results[0]})
    
    # Get the appropriate details list
    details = model.epoch_test_details if phase == "test" else model.epoch_validation_details
    
    # Save detailed results if we have any
    if details:
        csv_filename = f"{phase}_details{file_label}.csv"
        csv_path = model_dir / csv_filename
        model.log_to_csv(str(csv_path), details)
        logger.info(f"Saved {len(details)} {phase} results to {csv_path}")
    else:
        logger.warning(f"No {phase} details were collected - empty results")
    
    return results

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Ensure models_dir exists and is a Path object
    models_dir = Path(CONFIG["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique model directory with timestamp
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_dir = models_dir / f"pg_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logging(model_dir)
    
    # Initialize WandB - save to our model directory
    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False,
        save_dir=str(model_dir),  # str() for WandB compatibility
        config=convert_paths_to_strings(CONFIG)  # Convert Path objects before passing to WandB
    )
    
    # Save full config to model directory
    config_path = model_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(convert_paths_to_strings(CONFIG), f, indent=4)  # Convert Path objects before saving
    logger.info(f"Saved config to {config_path}")
    
    # Log experiment details
    logger.info(f"Starting experiment with config:\n{json.dumps(convert_paths_to_strings(CONFIG), indent=4)}")
    logger.info(f"All outputs will be saved to: {model_dir}")

    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        Path(CONFIG["data_dir"]),  # Ensure Path object
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )

    train_key, dev_key, test_key = (
        CONFIG["train_file"].split('.')[0],
        CONFIG["dev_file"].split('.')[0],
        CONFIG["test_file"].split('.')[0]
    )

    # Model setup - handle checkpoint path properly
    checkpoint_path = Path(CONFIG["pg_checkpoint_path"]) if CONFIG["pg_checkpoint_path"] else None
    model = setup_model(model_dir, "_pg", checkpoint_path)

    # Callbacks
    pg_checkpoint_callback = ModelCheckpoint(
        dirpath=str(model_dir),  # str() for PyTorch compatibility
        monitor='validation_pg_loss',
        mode='min',
        save_top_k=1,
        filename="pg_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss-{validation_pg_loss:.2f}",
        save_last=True,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='validation_pg_loss',
        min_delta=0.00,
        patience=2,
        verbose=True,
        mode='min'
    )

    # Trainer setup
    trainer = setup_trainer(
        CONFIG["pg_epochs"], 
        pg_checkpoint_callback, 
        early_stop_callback, 
        wandb_logger,
        model_dir
    )

    # Training
    logger.info("Starting training...")
    trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
    logger.info("Training completed.")

    # Load best model
    best_checkpoint = Path(pg_checkpoint_callback.best_model_path)
    logger.info(f"Loading best model from: {best_checkpoint}")
    model = setup_model(model_dir, "_pg", best_checkpoint)

    # Evaluation
    logger.info("Running validation...")
    val_results = evaluate_and_save(model, dataloaders[dev_key], "validation", model_dir, "_pg")

    logger.info("Running test...")
    test_results = evaluate_and_save(model, dataloaders[test_key], "test", model_dir, "_pg")

    # Finalize
    wandb.finish()
    logger.info(f"Experiment completed. All outputs saved to: {model_dir}")

if __name__ == '__main__':
    main()