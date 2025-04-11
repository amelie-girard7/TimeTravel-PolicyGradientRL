import sys
import os
import datetime
import logging
from pathlib import Path
import pytorch_lightning as pl
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from src.pg.models.model_ppo import FlanT5PPOFineTuner
from src.pg.data_loader import create_dataloaders
from src.pg.utils.configppo import CONFIG

# Setup basic logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger, swa_callback=None):
    """
    Configures and returns a PyTorch Lightning Trainer instance.
    """
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [cb for cb in [checkpoint_callback, early_stop_callback, lr_monitor, swa_callback] if cb is not None]

    return Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=0.25,
        accumulate_grad_batches=CONFIG.get("accumulate_grad_batches", 1),
        gradient_clip_val=CONFIG.get("gradient_clip_val", 0.5),
        gradient_clip_algorithm="norm",
        precision=CONFIG.get("precision", "32-true"),
        enable_progress_bar=True,
        overfit_batches=CONFIG.get("overfit_batches", 0),
        default_root_dir="./"
    )

def main():
    """
    Main function to prepare data, instantiate the PPO model, and start training.
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = Path(CONFIG["models_dir"]) / f"ppo_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        if not Path(CONFIG["data_dir"]).exists():
            raise FileNotFoundError(f"Data directory not found: {CONFIG['data_dir']}")

        # Initialize Weights & Biases logger.
        wandb_logger = WandbLogger(
            project="counterfactualStory",
            entity="counterfactualStory",
            log_model=False,
            save_dir=str(model_dir),
            config=CONFIG
        )

        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
        dataloaders = create_dataloaders(
            CONFIG["data_dir"],
            tokenizer,
            CONFIG["batch_size"],
            CONFIG["num_workers"],
        )

        # Configure checkpoint callback.
        ppo_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='val/avg_reward',
            mode='max',
            save_top_k=1,
            filename="ppo-epoch={epoch:02d}-step={step}-val_reward={val/avg_reward:.2f}-policy_loss={train/policy_loss:.2f}",
            save_weights_only=True,
            auto_insert_metric_name=False
        )

        # Early stopping is disabled for PPO.
        early_stop_callback = None

        # Early stopping callback to stop training when the validation loss stops improving
        # early_stop_callback = EarlyStopping(
        #     monitor='val/avg_reward',    # Metric to monitor
        #     min_delta=0.00,              # Minimum change to qualify as an improvement
        #     patience=2,                  # Number of epochs with no improvement to wait before stopping
        #     verbose=True,                # Enable verbose output for debugging
        #     mode='max'                   # Mode 'max' means we expect the metric to increase
        # )

        # Optional: Stochastic Weight Averaging callback.
        swa_callback = pl.callbacks.StochasticWeightAveraging(
            swa_lrs=1e-5,
            swa_epoch_start=0.7
        )

        model_args = {
            "model_name": CONFIG["model_name"],
            "model_dir": model_dir,
            "file_label": "_ppo"
        }

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

        trainer = setup_trainer(
            max_epochs=CONFIG["ppo_epochs"],
            checkpoint_callback=ppo_checkpoint_callback,
            early_stop_callback=early_stop_callback,
            wandb_logger=wandb_logger,
            swa_callback=swa_callback
        )

        # Start training.
        trainer.fit(
            model,
            train_dataloaders=dataloaders[CONFIG["train_file"].split('.')[0]],
            val_dataloaders=dataloaders[CONFIG["dev_file"].split('.')[0]],
            ckpt_path=CONFIG["ppo_checkpoint_path"] if CONFIG.get("ppo_resume_training") else None
        )

        # Test the trained model.
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
