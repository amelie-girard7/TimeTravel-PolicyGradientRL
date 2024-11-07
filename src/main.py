import os
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

    # Set model training mode
    model.use_policy_gradient = use_policy_gradient
    return model

def setup_trainer(model_dir, max_epochs, checkpoint_callback, wandb_logger, starting_epoch=0):
    """
    Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing.
    """
    trainer = Trainer(
        max_epochs=max_epochs + starting_epoch,  # total epochs from the start
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,  # Use the shared WandB logger instance
        callbacks=[checkpoint_callback],
        val_check_interval=0.1  # Set validation frequency
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs starting from epoch {starting_epoch}.")
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
        print("Starting MLE phase training...")
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

        # Train with MLE
        trainer = setup_trainer(model_dir, CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        
        # Capture the last completed epoch in MLE phase for continuity in PG phase
        last_mle_epoch = trainer.current_epoch
        print(f"MLE training completed. Last MLE epoch: {last_mle_epoch}")

    # --- PG Phase ---
    if CONFIG["pg_enabled"]:
        print("Starting PG phase training...")
        
        # Load from PG-specific checkpoint or latest MLE checkpoint
        pg_checkpoint = CONFIG["pg_checkpoint_path"] if CONFIG["pg_from_checkpoint"] else mle_checkpoint_callback.best_model_path
        model = setup_model(
            model_dir, 
            file_label="_pg", 
            checkpoint_path=pg_checkpoint, 
            use_policy_gradient=True
        )

        pg_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_pg_loss',
            mode='max',
            save_top_k=1,
            filename="best_pg_checkpoint"
        )

        # Set the starting epoch for PG phase to continue logging from MLE phase
        trainer = setup_trainer(model_dir, CONFIG["pg_epochs"], pg_checkpoint_callback, wandb_logger, starting_epoch=last_mle_epoch)
        
        # Start PG training from the MLE checkpoint
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key], ckpt_path=pg_checkpoint)
        print(f"PG training completed from epoch {last_mle_epoch} onward.")

    # --- Testing Phase ---
    if model:
        logger.info("Testing the final model.")
        trainer.test(model, dataloaders[test_key])
    else:
        logger.info("No model was trained, skipping testing.")

if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
