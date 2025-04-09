import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import T5Tokenizer
from src.pg.models.model_dpo import FlanT5DPOTrainer
from src.pg.data_loader_dpo import create_dataloaders
from src.pg.utils.config_dpo import CONFIG
import datetime
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Setup directories
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = Path(CONFIG["models_dir"]) / f"dpo_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer and dataloaders
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"])
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"]
    )
    
    # Initialize model
    model = FlanT5DPOTrainer(
        model_name=CONFIG["model_name"],
        model_dir=model_dir
    )
    
    # Setup trainer callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            monitor="val/alignment",
            mode="max",
            save_top_k=3,
            filename="dpo-{epoch}-{val/alignment:.2f}",
            auto_insert_metric_name=False
        ),
        EarlyStopping(
            monitor="val/alignment",
            patience=3,
            mode="max",
            min_delta=0.01,
            check_finite=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Initialize Wandb logger
    wandb_logger = WandbLogger(
        project="counterfactual-dpo",
        config=CONFIG,
        save_dir=str(model_dir),
        log_model=False
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=CONFIG.get("gradient_clip_val", 0.5),
        gradient_clip_algorithm="norm",
        val_check_interval=CONFIG.get("val_check_interval", 0.25),
        log_every_n_steps=10,
        deterministic=CONFIG.get("deterministic", False),
        precision=CONFIG.get("precision", "32-true")
    )
    
    try:
        # Train and test
        logger.info("Starting DPO training...")
        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["dev"]
        )
        
        logger.info("Starting final evaluation...")
        test_results = trainer.test(
            model,
            dataloaders=dataloaders["test"],
            ckpt_path="best"
        )
        logger.info(f"Test results: {test_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Training completed. Saving final model...")
        # Save the final model state
        final_path = model_dir / "final_model.ckpt"
        trainer.save_checkpoint(final_path)
        logger.info(f"Model saved to {final_path}")

if __name__ == "__main__":
    main()