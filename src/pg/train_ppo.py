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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger, swa_callback=None):
    """Configure and return the PyTorch Lightning Trainer"""
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    if swa_callback is not None:
        callbacks.append(swa_callback)
    
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
    try:
        # Setup environment and paths
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = Path(CONFIG["models_dir"]) / f"ppo_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Validate data directory
        if not Path(CONFIG["data_dir"]).exists():
            raise FileNotFoundError(f"Data directory not found: {CONFIG['data_dir']}")

        # Initialize WandB
        wandb_logger = WandbLogger(
            project="counterfactualStory",
            entity="counterfactualStory",
            log_model=False,
            save_dir=str(model_dir),
            config=CONFIG
        )

        # Prepare dataloaders
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
        dataloaders = create_dataloaders(
            CONFIG["data_dir"],
            tokenizer,
            CONFIG["batch_size"],
            CONFIG["num_workers"],
        )

        # Configure callbacks
        ppo_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='val/avg_reward',
            mode='max',
            save_top_k=3,
            filename="ppo-{epoch:02d}-{val/avg_reward:.2f}",
            save_weights_only=True,
            auto_insert_metric_name=False
        )

        early_stop_callback = EarlyStopping(
            monitor='val/avg_reward',
            patience=3,
            mode='max',
            min_delta=0.05,
            check_finite=True,
            stopping_threshold=1.0
        )

        # Add SWA for stability
        swa_callback = pytorch_lightning.callbacks.StochasticWeightAveraging(
            swa_lrs=1e-5,
            swa_epoch_start=0.7
        )

        # Model initialization
        model_args = {
            "model_name": CONFIG["model_name"],
            "model_dir": model_dir,
            "file_label": "_ppo"
        }

        if CONFIG.get("ppo_from_checkpoint", False):
            logger.info(f"Loading weights from: {CONFIG['ppo_checkpoint_path']}")
            model = FlanT5PPOFineTuner.load_from_checkpoint(
                CONFIG["ppo_checkpoint_path"],
                **model_args,
                strict=False
            )
            logger.info("Successfully loaded checkpoint weights")
        else:
            model = FlanT5PPOFineTuner(**model_args)
            logger.info("Initialized new PPO model")

        # Setup and run trainer (now with SWA)
        trainer = setup_trainer(
            max_epochs=CONFIG["ppo_epochs"],
            checkpoint_callback=ppo_checkpoint_callback,
            early_stop_callback=early_stop_callback,
            wandb_logger=wandb_logger,
            swa_callback=swa_callback  # Pass SWA here
        )

        # Training
        trainer.fit(
            model,
            train_dataloaders=dataloaders[CONFIG["train_file"].split('.')[0]],
            val_dataloaders=dataloaders[CONFIG["dev_file"].split('.')[0]],
            ckpt_path=CONFIG["ppo_checkpoint_path"] if CONFIG.get("ppo_resume_training") else None
        )

        # Testing
        test_results = trainer.test(
            model,
            dataloaders=dataloaders[CONFIG["test_file"].split('.')[0]],
            ckpt_path="best"
        )
        logger.info(f"Final test results: {test_results}")

    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Training session completed")

if __name__ == '__main__':
    logger.info("Starting PPO training pipeline...")
    main()