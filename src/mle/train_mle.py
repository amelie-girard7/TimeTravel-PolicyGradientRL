import os
import sys
import datetime
from pathlib import Path
import logging
import torch
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.mle.models.model_T5 import FlanT5FineTuner
from src.mle.data_loader import create_dataloaders
from src.mle.utils.config import CONFIG

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
    data_path = CONFIG["data_dir"] / 'transformed'
    batch_size = CONFIG["batch_size"]
    num_workers = CONFIG["num_workers"]
    dataloaders = create_dataloaders(data_path, model.tokenizer, batch_size, num_workers)
    return dataloaders

def setup_trainer(model_dir, wandb_logger):
    """
    Configures the training environment with checkpoints and logging.
    """
    logger.info("Setting up the trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_on_train_epoch_end=False
    )

    trainer = Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,  # Only using W&B logger
    )
    return trainer

def main():
    """
    Main function orchestrating the model training and evaluation process.
    """
    # Set the GPU manually
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        # Timestamp for unique directory creation
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = CONFIG["models_dir"] / f"model_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Simple W&B logger setup - only tracking losses
        wandb_logger = WandbLogger(
            project="counterfactualStory",
            name=f"mle_{model_timestamp}",
            save_dir=str(model_dir),
            config=CONFIG
        )

        logger.info("Tokenizer setup...")
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)

        logger.info("Model setup...") 
        model = setup_model(model_dir)

        logger.info("Dataloaders setup...")
        dataloaders = setup_dataloaders(model, tokenizer)

        logger.info("Trainer setup...")
        trainer = setup_trainer(model_dir, wandb_logger)
        
        # Extract dataset keys
        train_key = CONFIG["train_file"].split('.')[0]
        dev_key = CONFIG["dev_file"].split('.')[0]
        test_key = CONFIG["test_file"].split('.')[0]
        
        logger.info("Starting training...")
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])

        logger.info("Starting testing...")
        trainer.test(model, dataloaders[test_key])

    except Exception as e:
        logger.exception("An unexpected error occurred during the process.")
        sys.exit(1)

if __name__ == '__main__':
    logger.info("Starting the main process...") 
    main()
    logger.info("Process completed.")