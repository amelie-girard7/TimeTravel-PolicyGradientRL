# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/main.py
import os
import sys
import datetime
import logging
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.models.model import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.config import CONFIG

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir, file_label="", checkpoint_path=None, use_policy_gradient=False):
    """
    Initializes the model, optionally loading from a checkpoint.
    """
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )
    else:
        logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(CONFIG["model_name"], model_dir, file_label=file_label)

    model.use_policy_gradient = use_policy_gradient
    return model

def setup_trainer(model_dir, max_epochs, checkpoint_callback, wandb_logger):
    """
    Sets up the PyTorch Lightning Trainer with WandB logger and checkpointing.
    """
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,  # Use the shared WandB logger instance
        callbacks=[checkpoint_callback],
        val_check_interval=0.1
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Unique directory based on timestamp
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"experiment_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model="all"
    )
    wandb_logger.experiment.config.update(CONFIG)
    
    # Setup tokenizer and dataloaders
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(CONFIG["data_dir"], tokenizer, CONFIG["batch_size"], CONFIG["num_workers"])
    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], CONFIG["test_file"].split('.')[0]

    model = None  # Initialize model variable to track if any phase trained a model

    # --- MLE Phase ---
    if CONFIG["mle_enabled"]:
        mle_checkpoint = CONFIG["mle_checkpoint_path"] if CONFIG["mle_from_checkpoint"] else None
        model = setup_model(
            model_dir, 
            file_label="_mle", 
            checkpoint_path=mle_checkpoint, 
            use_policy_gradient=False
        )

        mle_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_mle_loss',
            mode='min',
            save_top_k=1,
            filename="best_mle_checkpoint"
        )
        trainer = setup_trainer(model_dir, CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])

    # --- PG Phase ---
    if CONFIG["pg_enabled"]:
        # Choose the checkpoint based on config:
        # - If "pg_from_checkpoint" is True, use "pg_checkpoint_path" if provided.
        # - If no path is provided, use the best MLE checkpoint (from the prior MLE phase).
        pg_checkpoint = CONFIG["pg_checkpoint_path"] if CONFIG["pg_from_checkpoint"] else mle_checkpoint_callback.best_model_path
        
        # Set up the model for PG, either from the checkpoint or fresh from Hugging Face
        model = setup_model(
            model_dir, 
            file_label="_pg", 
            checkpoint_path=pg_checkpoint,  # This can be None if starting fresh
            use_policy_gradient=True        # Set to True to use PG mode
        )

        # Optional: If MLE was trained beforehand, continue the epoch count
        model.current_epoch = CONFIG["mle_epochs"] if CONFIG["mle_enabled"] else 0

        pg_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_pg_loss',
            mode='max',
            save_top_k=1,
            filename="best_pg_checkpoint"
        )
        trainer = setup_trainer(model_dir, CONFIG["pg_epochs"], pg_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])

    # --- Testing Phase ---
    if model:
        # Only test if a model was trained (either MLE or PG)
        logger.info("Testing the final model.")
        trainer.test(model, dataloaders[test_key])
    else:
        logger.info("No model was trained, skipping testing.")

if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
