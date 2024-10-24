# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/main_t5.py
import os
import sys
import datetime
from pathlib import Path
import logging
import torch
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.models.model_T5 import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.config import CONFIG

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir):
    """
    Prepares the FlanT5FineTuner model for training. 
    This includes loading from the provided checkpoint.
    """
    # Load the pre-trained model from the specified checkpoint if available
    checkpoint_path = CONFIG.get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path, model_name=CONFIG["model_name"], model_dir=model_dir
        )
    else:
        logger.info(f"Initializing model from pre-trained: {CONFIG['model_name']}")
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

    dataloaders = create_dataloaders(data_path, tokenizer, batch_size, num_workers)
    return dataloaders

def setup_trainer(model,model_dir):
    """
    Configures the training environment, including checkpoints, logging, and GPU setup.
    """
    logger.info("Setting up the trainer...")

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project="counterfactual_story_rewriting",
        entity="ACL-paper",  # Team 
        log_model="all"
    )

    # Log configuration parameters (CONFIG) to W&B
    wandb_logger.experiment.config.update(CONFIG)

    # Watch the model to log gradients and weights (for tracking the computational graph)
    wandb_logger.watch(model, log="all")

    # Create checkpoint callbacks for both policy gradient loss and reward
    checkpoint_loss_callback = ModelCheckpoint(
        dirpath=model_dir,
        every_n_train_steps=100,  
        save_top_k=1,
        monitor='validation_policy_gradient_loss',
        mode='min',
    )

    checkpoint_reward_callback = ModelCheckpoint(
        dirpath=model_dir,
        every_n_train_steps=100,  
        save_top_k=1,
        monitor='validation_policy_gradient_reward',
        mode='max',
    )

    # Set up the Trainer
    trainer = Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator='gpu',  
        devices=1,
        callbacks=[checkpoint_loss_callback, checkpoint_reward_callback],
        logger=[wandb_logger],
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
        model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directories exist
        
        logger.info("Tokenizer setup...")
        # Setup tokenizer
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)

        logger.info("Model setup...") 
        # Prepare model, dataloaders, and trainer
        model = setup_model(model_dir)

        logger.info("Model initialized, about to load dataloaders...") 
        # Setup dataloaders
        dataloaders = setup_dataloaders(model, tokenizer)

        logger.info("Dataloaders created, about to set up the trainer...")
        # Pass the model into setup_trainer
        trainer = setup_trainer(model,model_dir)
         
        # Extract the keys for train, dev, and test from CONFIG and remove the file extension
        train_key = CONFIG["train_file"].split('.')[0]  # 'train_supervised_small'
        dev_key = CONFIG["dev_file"].split('.')[0]      # 'dev_data'
        test_key = CONFIG["test_file"].split('.')[0]    # 'test_data'
        
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
