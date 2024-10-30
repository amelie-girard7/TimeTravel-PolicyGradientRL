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

def setup_model(model_dir, file_label=""):
    """
    Initializes the T5 model from Hugging Face for training with unique CSV file paths.
    """
    logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
    model = FlanT5FineTuner(CONFIG["model_name"], model_dir, file_label=file_label)
    return model

def setup_trainer(model_dir, max_epochs, checkpoint_callback, wandb_project_name="counterfactual_story_rewriting"):
    """
    Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing.
    """
    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        entity="ACL-paper",
        log_model="all"
    )
    wandb_logger.experiment.config.update(CONFIG)  # Sync CONFIG with WandB

    # Initialize Trainer with W&B logging, checkpointing, and validation interval
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1  # Validate every 10% of an epoch
    )
    return trainer

def main():
    """
    Main function for orchestrating two training experiments: one with only MLE and another with MLE + PG.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU

    try:
        # Timestamp for unique directory creation
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir_model1 = CONFIG["models_dir"] / f"model1_{model_timestamp}"
        model_dir_model2 = CONFIG["models_dir"] / f"model2_{model_timestamp}"
        model_dir_model1.mkdir(parents=True, exist_ok=True)
        model_dir_model2.mkdir(parents=True, exist_ok=True)
        
        # Setup tokenizer and dataloaders
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
        dataloaders = create_dataloaders(CONFIG["data_dir"], tokenizer, CONFIG["batch_size"], CONFIG["num_workers"])
        train_key, dev_key, test_key = (
            CONFIG["train_file"].split('.')[0], 
            CONFIG["dev_file"].split('.')[0], 
            CONFIG["test_file"].split('.')[0]
        )

        # --- Train Model 1: MLE for 6 epochs ---
        logger.info("Training Model 1 with MLE for 6 epochs...")
        model1 = setup_model(model_dir_model1, file_label="_mle_model1")
        model1.use_policy_gradient = False  # MLE training mode
        mle_checkpoint_callback_model1 = ModelCheckpoint(
            dirpath=model_dir_model1,
            monitor='validation_mle_loss',
            mode='min',
            save_top_k=1,
            save_last=True,  # Always save the last model
            filename="best_mle_checkpoint_model1"
        )
        mle_trainer_model1 = setup_trainer(model_dir_model1, CONFIG["mle_epochs_model1"], mle_checkpoint_callback_model1)
        mle_trainer_model1.fit(model1, dataloaders[train_key], dataloaders[dev_key])
        
        # --- Train Model 2: MLE for 3 epochs ---
        logger.info("Training Model 2 with MLE for 3 epochs...")
        model2 = setup_model(model_dir_model2, file_label="_mle_model2")
        model2.use_policy_gradient = False  # MLE training mode
        mle_checkpoint_callback_model2 = ModelCheckpoint(
            dirpath=model_dir_model2,
            monitor='validation_mle_loss',
            mode='min',
            save_top_k=1,
            save_last=True,  # Always save the last model
            filename="best_mle_checkpoint_model2"
        )
        mle_trainer_model2 = setup_trainer(model_dir_model2, CONFIG["mle_epochs_model2"], mle_checkpoint_callback_model2)
        mle_trainer_model2.fit(model2, dataloaders[train_key], dataloaders[dev_key])

        # Retrieve the best MLE checkpoint path for Model 2
        best_mle_checkpoint_model2 = mle_checkpoint_callback_model2.best_model_path or mle_checkpoint_callback_model2.last_model_path
        if not best_mle_checkpoint_model2:
            logger.error("No checkpoint was saved during MLE training for Model 2.")
            sys.exit(1)
        
        logger.info(f"Loading the best MLE model for Model 2 from {best_mle_checkpoint_model2}")
        
        # --- Fine-tune Model 2 with Policy Gradient for 3 epochs ---
        model2_pg = FlanT5FineTuner.load_from_checkpoint(
            best_mle_checkpoint_model2,
            model_name=CONFIG["model_name"],
            model_dir=model_dir_model2,
            file_label="_pg_model2"
        )
        model2_pg.use_policy_gradient = True  # PG training mode
        pg_checkpoint_callback_model2 = ModelCheckpoint(
            dirpath=model_dir_model2,
            monitor='validation_policy_gradient_reward_mean',
            mode='max',
            save_top_k=1,
            save_last=True,  # Always save the last model
            filename="best_pg_checkpoint_model2"
        )
        pg_trainer_model2 = setup_trainer(model_dir_model2, CONFIG["pg_epochs_model2"], pg_checkpoint_callback_model2)
        pg_trainer_model2.fit(model2_pg, dataloaders[train_key], dataloaders[dev_key])

        # --- Testing Phase for both models ---
        logger.info("Testing Model 1 (MLE 6 epochs)...")
        mle_trainer_model1.test(model1, dataloaders[test_key])
        
        logger.info("Testing Model 2 (MLE 3 epochs + PG 3 epochs)...")
        pg_trainer_model2.test(model2_pg, dataloaders[test_key])

    except Exception as e:
        logger.exception("An unexpected error occurred during the process.")
        sys.exit(1)

if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
