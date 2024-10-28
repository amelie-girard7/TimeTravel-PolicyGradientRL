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
    Initializes the T5 model. 
    If `CONFIG["use_checkpoint"]` is True and `CONFIG["checkpoint_path"]` exists, load from that checkpoint.
    Otherwise, download the model from Hugging Face.
    """
    checkpoint_path = CONFIG.get("checkpoint_path")
    
    if CONFIG["use_checkpoint"] and checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path, model_name=CONFIG["model_name"], model_dir=model_dir
        )
    else:
        logger.info(f"No valid checkpoint found or requested. Initializing a fresh model: {CONFIG['model_name']}")
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

def setup_trainer(model, model_dir):
    logger.info("Setting up the trainer...")

    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project="counterfactual_story_rewriting",
        entity="ACL-paper",  
        log_model="all"
    )
    wandb_logger.experiment.config.update(CONFIG)

    # Dynamically choose checkpoint callbacks based on the training mode
    if CONFIG["use_policy_gradient"]:
        logger.info("Using policy gradient mode, setting up checkpoints for policy gradient.")
        
        # Create checkpoints for policy gradient loss and reward
        checkpoint_loss_callback = ModelCheckpoint(
            dirpath=model_dir, 
            every_n_train_steps=100,
            save_top_k=1,
            monitor='validation_policy_gradient_loss',  # Monitor policy gradient loss
            mode='min'
        )
        checkpoint_reward_mean_callback = ModelCheckpoint(
            dirpath=model_dir,
            every_n_train_steps=100,
            save_top_k=1,# Only keep the best checkpoint
            monitor='validation_policy_gradient_reward_mean',  # Monitor policy gradient reward
            mode='max'
        )
        callbacks = [checkpoint_loss_callback, checkpoint_reward_mean_callback]
    
    else:
        logger.info("Using MLE mode, setting up checkpoints for MLE loss.")
        
        # Create checkpoints for MLE loss
        checkpoint_mle_loss_callback  = ModelCheckpoint(
            dirpath=model_dir, 
            every_n_train_steps=100,
            save_top_k=1,
            monitor='validation_mle_loss',  # Monitor MLE loss
            mode='min'
        )
        checkpoint_mle_score_callback = ModelCheckpoint(
            dirpath=model_dir,
            every_n_train_steps=100,
            save_top_k=1,
            monitor='validation_mle_score_mean',  # Monitor MLE score mean
            mode='max'
        )
        callbacks = [checkpoint_mle_loss_callback, checkpoint_mle_score_callback]

    # Set up the trainer with the selected callbacks
    trainer = Trainer(
        max_epochs=CONFIG["max_epochs"], 
        accelerator='gpu',  
        devices=1,
        callbacks=callbacks,  # Use the dynamically selected callbacks
        logger=[wandb_logger],
        val_check_interval=0.1
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
