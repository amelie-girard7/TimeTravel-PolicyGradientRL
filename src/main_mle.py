# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/main_mle.py
import os
import sys
import datetime
from pathlib import Path
import logging
import torch
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # Updated import for W&B
from src.models.model_mle import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.config import CONFIG

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir):
    """
    Prepares the FlanT5FineTuner model for training.
    """
    model = FlanT5FineTuner(CONFIG["model_name"], model_dir)
    return model

def setup_dataloaders(model, tokenizer):
    """
    Creates dataloaders for training, validation, and testing phases.
    """
    logger.info("Setting up dataloaders...")
    data_path = CONFIG["data_dir"]

    batch_size = CONFIG["batch_size"]
    num_workers = CONFIG["num_workers"]

    dataloaders = create_dataloaders(data_path, model.tokenizer, batch_size, num_workers)
    return dataloaders

def setup_trainer(model_dir, max_epochs_mle, wandb_project_name="counterfactual_story_rewriting"):
    """
    Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing.
    """
    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        #every_n_train_steps=100,
        filename='checkpoint-{epoch:02d}-{mle_val_loss:.2f}',  # Naming includes validation loss in filename
        monitor='mle_val_loss',  # Checkpoints saved based on mle_val_loss improvement
        mode='min',
        save_top_k=1,  # Only keep the best model based on validation loss
    )

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        entity="ACL-paper",
        log_model="all"
    )
    wandb_logger.experiment.config.update(CONFIG)  # Sync CONFIG with WandB

    # Initialize Trainer with W&B logging, checkpointing, and validation interval
    trainer = Trainer(
        max_epochs=max_epochs_mle,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1  # Validate every 100 training steps
    )
    return trainer

def main():
    """
    Main function orchestrating the model training and evaluation process.
    """
    # Set the GPU manually
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Make sure only one GPU is used

    try:
        # Timestamp for unique directory creation
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = CONFIG["models_dir"] / f"model_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Tokenizer setup...")
        # Setup tokenizer
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)

        logger.info("Model setup...")
        # Prepare model, dataloaders, and trainer
        model = setup_model(model_dir)

        logger.info("Model initialized, about to load dataloaders.")
        # Setup dataloaders
        dataloaders = setup_dataloaders(model, tokenizer)

        logger.info("Dataloaders created, about to set up the trainer.")
        # Setup trainer with W&B logger
        trainer = setup_trainer(model_dir, CONFIG["max_epochs_mle"])

        # Extract the keys for train, dev, and test from CONFIG and remove the file extension
        train_key = CONFIG["train_file"].split('.')[0]
        dev_key = CONFIG["dev_file"].split('.')[0]
        test_key = CONFIG["test_file"].split('.')[0]

        logger.info("Trainer setup complete, starting training...")

        # Start training
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])

        logger.info("Training complete, starting testing...")

        # Start testing
        trainer.test(model, dataloaders[test_key])

    except Exception as e:
        logger.exception("An unexpected error occurred during the process.")
        sys.exit(1)

if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
