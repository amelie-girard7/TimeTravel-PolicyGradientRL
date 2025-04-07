# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/train_ppo.py
import sys
import os
import datetime
import logging
from pathlib import Path
import pytorch_lightning
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.pg.models.model_ppo import FlanT5PPOFineTuner
from src.pg.data_loader import create_dataloaders
from src.pg.utils.configppo import CONFIG

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Timestamp, logger name, log level, and message
)
logger = logging.getLogger(__name__)


def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger, swa_callback=None):
    """
    Configure and return a PyTorch Lightning Trainer instance.
    This setup ensures safe handling of optional callbacks (e.g., early stopping or SWA).
    """

    # Callback to monitor learning rate at each step
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Compose the list of callbacks by filtering out None values
    # This prevents errors from PyTorch Lightning trying to use .on_exception() on None
    callbacks = [cb for cb in [checkpoint_callback, early_stop_callback, lr_monitor, swa_callback] if cb is not None]

    return Trainer(
        max_epochs=max_epochs,  # Maximum number of training epochs
        accelerator='gpu',      # Use GPU for training
        devices=1,              # Number of GPUs to use
        logger=wandb_logger,    # Use Weights & Biases for experiment tracking
        callbacks=callbacks,    # Safe, non-None list of callbacks
        val_check_interval=0.25,  # Run validation 4 times per epoch
        accumulate_grad_batches=CONFIG.get("accumulate_grad_batches", 1),  # Gradient accumulation to simulate large batch
        gradient_clip_val=CONFIG.get("gradient_clip_val", 0.5),            # Clip gradients to prevent exploding gradients
        gradient_clip_algorithm="norm",                                    # Use L2 norm for gradient clipping
        precision=CONFIG.get("precision", "32-true"),                      # Mixed precision training (can be "16-mixed" etc.)
        enable_progress_bar=True,         # Show training progress bar
        overfit_batches=CONFIG.get("overfit_batches", 0),  # For debugging: overfit a few batches
        default_root_dir="./"             # Root path for saving logs/checkpoints
    )



def main():
    """
    Main function to set up and execute PPO training.
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Explicitly set the GPU device to be used

        # Create unique directory for the current training session
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = Path(CONFIG["models_dir"]) / f"ppo_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Check if data directory exists
        if not Path(CONFIG["data_dir"]).exists():
            raise FileNotFoundError(f"Data directory not found: {CONFIG['data_dir']}")

        # Initialize Weights & Biases logger
        wandb_logger = WandbLogger(
            project="counterfactualStory",
            entity="counterfactualStory",
            log_model=False,
            save_dir=str(model_dir),
            config=CONFIG
        )

        # Load tokenizer and create data loaders
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
        dataloaders = create_dataloaders(
            CONFIG["data_dir"],
            tokenizer,
            CONFIG["batch_size"],
            CONFIG["num_workers"],
        )

        # Configure checkpoint callback
        ppo_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='val/avg_reward',
            mode='max',
            save_top_k=3,
            filename="ppo-epoch={epoch:02d}-step={step}-val_reward={val/avg_reward:.2f}-policy_loss={train/policy_loss:.2f}",
            save_weights_only=True,
            auto_insert_metric_name=False
        )

        # Configure early stopping callback
        # early_stop_callback = EarlyStopping(
        #     monitor='val/avg_reward',  # Metric to monitor
        #     patience=3,  # Stop after 3 epochs without improvement
        #     mode='max',  # Maximize monitored metric
        #     min_delta=0.05,  # Minimum change to qualify as improvement
        #     check_finite=True,
        #     stopping_threshold=1.0  # Threshold to trigger immediate stop
        # )

        early_stop_callback = None  # Disable early stopping explicitly

        # Stochastic Weight Averaging for improved training stability
        swa_callback = pytorch_lightning.callbacks.StochasticWeightAveraging(
            swa_lrs=1e-5,  # Learning rate for SWA
            swa_epoch_start=0.7  # Start SWA after 70% of epochs
        )

        # Arguments for PPO model initialization
        model_args = {
            "model_name": CONFIG["model_name"],
            "model_dir": model_dir,
            "file_label": "_ppo"
        }

        # Initialize or load PPO model from checkpoint
        if CONFIG.get("ppo_from_checkpoint", False):
            logger.info(f"Loading pretrained weights from: {CONFIG['ppo_checkpoint_path']}")
            model = FlanT5PPOFineTuner.load_from_checkpoint(
                CONFIG["ppo_checkpoint_path"],
                **model_args,
                strict=False
            )
            logger.info("Checkpoint loaded successfully.")
        else:
            model = FlanT5PPOFineTuner(**model_args)
            logger.info("Initialized new PPO model from scratch.")

        # Setup trainer
        trainer = setup_trainer(
            max_epochs=CONFIG["ppo_epochs"],
            checkpoint_callback=ppo_checkpoint_callback,
            early_stop_callback=early_stop_callback,
            wandb_logger=wandb_logger,
            swa_callback=swa_callback
        )

        # Start training
        trainer.fit(
            model,
            train_dataloaders=dataloaders[CONFIG["train_file"].split('.')[0]],
            val_dataloaders=dataloaders[CONFIG["dev_file"].split('.')[0]],
            ckpt_path=CONFIG["ppo_checkpoint_path"] if CONFIG.get("ppo_resume_training") else None
        )

        # Test trained model
        test_results = trainer.test(
            model,
            dataloaders=dataloaders[CONFIG["test_file"].split('.')[0]],
            ckpt_path="best"
        )
        logger.info(f"Final test results: {test_results}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Training completed.")


if __name__ == '__main__':
    logger.info("Starting PPO training...")
    main()