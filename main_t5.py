# Timetravel/main.py
import os
import sys
import datetime
from pathlib import Path
import logging
import tokenizers
import torch
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.model_T5 import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir):
    """
    Prepares the FlanT5FineTuner model for training.

    Parameters:
        model_dir (Path): The directory to save model checkpoints.

    Returns:
        An instance of FlanT5FineTuner configured for training.
    """
    model = FlanT5FineTuner(CONFIG["model_name"], model_dir)
    return model

def setup_dataloaders(model, tokenizer):
    """
    Creates dataloaders for training, validation, and testing phases.

    Parameters:
        model: The model instance for which dataloaders are being prepared.
               Not directly used in this function but can be utilized for
               model-specific data adjustments.
        tokenizer: The tokenizer to process text data for the model.

    Returns:
        A dictionary with keys 'train', 'val', and 'test' pointing to their respective dataloaders.
    """
    data_path = CONFIG["data_dir"] / 'transformed'
    print(f"Data path: {data_path}")  # Debug print to ensure the path is correct

    batch_size = CONFIG["batch_size"]
    num_workers = CONFIG["num_workers"]

    dataloaders = create_dataloaders(data_path, model.tokenizer, batch_size, num_workers)
    return dataloaders

def setup_trainer(model_dir):
    """
    Configures the training environment with checkpoints and logging.

    Parameters:
        model_dir (Path): The directory for saving checkpoints and logs.

    Returns:
        Configured PyTorch Lightning Trainer.
    """

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    
    tensorboard_logger  = TensorBoardLogger(save_dir=model_dir, name="training_logs")
    trainer = Trainer(
            max_epochs=CONFIG["max_epochs"],
            accelerator='gpu' if torch.cuda.is_available() else None,
            #devices='auto' if torch.cuda.is_available() else None,
            devices=2 if torch.cuda.is_available() else None,  # Use 2 GPUs
            strategy='ddp',  # Use Distributed Data Parallel strategy for multi-GPU
            callbacks=[checkpoint_callback],
            logger=tensorboard_logger,
            #num_sanity_val_steps=0
            #accumulate_grad_batches=2,  # Adjust this based on your memory constraints
        )
    return trainer

def main():
    """
    Main function orchestrating the model training and evaluation process.
    """
    # Set the GPU manually
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # Make both GPUs available


    # Set Tensor Core precision policy for better performance on Tensor Cores
    # torch.set_float32_matmul_precision('high')

    try:
        # Timestamp for unique directory creation
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = CONFIG["models_dir"] / f"model_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True) # Ensure directories exist
        
        # Setup tokenizer
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
        
        # Prepare model, dataloaders, and trainer
        model = setup_model(model_dir)
        dataloaders = setup_dataloaders(model, tokenizer)
        trainer = setup_trainer(model_dir)
        
        # Extract the keys for train, dev, and test from CONFIG and remove the file extension
        train_key = CONFIG["train_file"].split('.')[0]  # 'train_supervised_small_sample'
        dev_key = CONFIG["dev_file"].split('.')[0]      # 'dev_data_sample'
        test_key = CONFIG["test_file"].split('.')[0]    # 'test_data_sample'
        
        # Start training
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        
        # Start testing
        trainer.test(model, dataloaders[test_key])

    except Exception as e:
        logger.exception("An unexpected error occurred during the process.")
        sys.exit(1)

if __name__ == '__main__':
    main()