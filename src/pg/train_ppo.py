import os
# allow CUDA allocator to expand its segments instead of fragmenting
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
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

# Console logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger):
    """
    Configures and returns a PyTorch Lightning Trainer instance.
    """
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [cb for cb in [checkpoint_callback, early_stop_callback, lr_monitor] if cb is not None]

    return Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=0.25,
        accumulate_grad_batches=CONFIG.get("accumulate_grad_batches", 1),
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

        # ─── File Logging ──────────────────────────────────────────────────────────
        # append DEBUG+ logs to a file in model_dir
        log_file = model_dir / "training_ppo.log"
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # attach to root logger so all library logs also go into the file
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to console and to '{log_file}'")

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
            filename="ppo-epoch={epoch:02d}-step={step}-val_reward={val/avg_reward:.2f}-ppo_loss={train/ppo_loss:.2f}",
            save_weights_only=True,
            auto_insert_metric_name=False
        )

        # Early stopping is disabled for PPO.
        early_stop_callback = None

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

        # Debug tokenizer behavior
        logger.debug("Running tokenizer diagnostics...")
        test_samples = [
            "The quick brown fox",
            "Sample with numbers: 12345",
            "Special chars: !@#$%^&*()",
            "New\nlines\tand\tspaces"
        ]

        for text in test_samples:
            try:
                encoded = model.tokenizer.encode(text, add_special_tokens=False)
                decoded = model.tokenizer.decode(encoded)

                logger.debug(
                    f"Tokenizer Test:\n"
                    f"Original: '{text}'\n"
                    f"Encoded IDs: {encoded}\n"
                    f"Decoded: '{decoded}'\n"
                    f"Lengths: {len(text)} chars → {len(encoded)} tokens\n"
                    f"Matches: {model._compare_texts(text, decoded)}"
                )

                batch = model.tokenizer([text]*2, return_tensors='pt', padding=True)
                logger.debug(
                    f"Batch Input IDs shape: {batch['input_ids'].shape}\n"
                    f"Attention mask: {batch['attention_mask'].tolist()}"
                )

            except Exception as e:
                logger.error(f"Tokenizer failed on text: '{text}'\nError: {str(e)}")





        trainer = setup_trainer(
            max_epochs=CONFIG["training_epochs"],
            checkpoint_callback=ppo_checkpoint_callback,
            early_stop_callback=early_stop_callback,
            wandb_logger=wandb_logger
        )

        # Start training.
        trainer.fit(
            model,
            train_dataloaders=dataloaders[CONFIG["train_file"].split('.')[0]],
            val_dataloaders=dataloaders[CONFIG["dev_file"].split('.')[0]],
            ckpt_path=CONFIG["ppo_checkpoint_path"] if CONFIG.get("ppo_resume_training") else None
        )

        # Test the trained model.
        best_ckpt_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.save_top_k and cb.monitor:
                best_ckpt_path = cb.best_model_path
                break

        if best_ckpt_path:
            print(f"Testing using best checkpoint: {best_ckpt_path}")
            trainer.test(
                model,
                dataloaders=dataloaders[CONFIG["test_file"].split('.')[0]],
                ckpt_path=best_ckpt_path
            )
        else:
            print("No best checkpoint found; testing on current weights.")
            trainer.test(
                model,
                dataloaders=dataloaders[CONFIG["test_file"].split('.')[0]]
            )

    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        logger.info("Training completed.")

if __name__ == '__main__':
    logger.info("Starting PPO training...")
    main()
