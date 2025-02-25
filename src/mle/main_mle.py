# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/mle/main_mle.py
import sys
import os
import datetime
import logging
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.mle.models.model import FlanT5FineTuner
from src.mle.data_loader import create_dataloaders
from src.mle.utils.metrics import MetricsEvaluator
from src.mle.utils.config import CONFIG
import pandas as pd
import re

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# Model Setup Functions
# ==============================

def setup_model(model_dir, file_label="", checkpoint_path=None):
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
    return model


def setup_trainer(max_epochs, checkpoint_callback, wandb_logger):
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer


# ==============================
# Evaluation and Metrics
# ==============================

def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
    logger.info(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")

    # Load model from checkpoint
    model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)

    # Perform evaluation
    trainer = Trainer(accelerator='gpu', devices=1)
    if phase == "test":
        trainer.test(model, loader, verbose=False)
    elif phase == "validation":
        trainer.validate(model, loader, verbose=False)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Load details for evaluation
    details_file = os.path.join(model_dir, f"{phase}_details{file_label}.csv")
    if not os.path.exists(details_file):
        logger.error(f"{phase.capitalize()} details file not found at {details_file}")
        raise FileNotFoundError(f"{phase.capitalize()} details file not found at {details_file}")

    details_df = pd.read_csv(details_file)
    filtered_details = details_df[details_df['Epoch'] == best_epoch]
    if filtered_details.empty:
        logger.warning(f"No rows found for epoch {best_epoch}. Using all rows for metrics calculation.")
        filtered_details = details_df

    # Extract relevant columns
    try:
        generated_texts = filtered_details['Generated Text'].tolist()
        edited_endings = filtered_details['Edited Ending'].tolist()
    except KeyError as e:
        logger.error(f"Missing column in details file: {e}")
        raise

    # Calculate metrics
    evaluator = MetricsEvaluator()
    metrics = {}

    try:
        metrics.update(evaluator.calculate_and_log_bart_similarity(generated_texts, edited_endings, logger))
        metrics.update(evaluator.calculate_and_log_bert_similarity(generated_texts, edited_endings, logger))
        metrics.update(evaluator.calculate_and_log_bleu_scores(generated_texts, edited_endings, logger))
        metrics.update(evaluator.calculate_and_log_rouge_scores(generated_texts, edited_endings, logger))
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics for {phase}: {e}")

    # Save metrics
    metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    metrics_df.to_csv(metrics_file, index=False)

    logger.info(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")


def extract_epoch_from_checkpoint(checkpoint_path):
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    logger.warning(f"Could not extract epoch from checkpoint path: {checkpoint_path}")
    return "Unknown"


# ==============================
# Main Training Function
# ==============================

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Create a unique directory based on timestamp
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"mle_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False
    )
    wandb_logger.experiment.config.update(CONFIG)

    # Setup tokenizer and dataloaders
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"]
    )

    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], CONFIG["test_file"].split('.')[0]

    # Train MLE Model
    print("Starting MLE phase training...")
    mle_checkpoint = CONFIG["mle_checkpoint_path"] if CONFIG["mle_from_checkpoint"] else None
    model = setup_model(
        model_dir,
        file_label="_mle",
        checkpoint_path=mle_checkpoint
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        monitor='validation_mle_loss',
        mode='min',
        save_top_k=1,
        filename="mle_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss={validation_mle_loss:.2f}"
    )

    trainer = setup_trainer(CONFIG["mle_epochs"], checkpoint_callback, wandb_logger)
    trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
    print("MLE training completed.")

    # Get best checkpoint and evaluate
    best_checkpoint = checkpoint_callback.best_model_path
    best_loss = checkpoint_callback.best_model_score
    logger.info(f"Best MLE checkpoint: {best_checkpoint}")
    logger.info(f"Best Validation MLE Loss: {best_loss:.4f}")

    if best_checkpoint:
        best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
        logger.info(f"Best checkpoint corresponds to epoch: {best_epoch}")

        evaluate_and_save(
            model_dir=model_dir,
            loader=dataloaders[test_key],
            best_checkpoint=best_checkpoint,
            file_label="_mle",
            best_epoch=best_epoch,
            phase="test"
        )
        evaluate_and_save(
            model_dir=model_dir,
            loader=dataloaders[dev_key],
            best_checkpoint=best_checkpoint,
            file_label="_mle",
            best_epoch=best_epoch,
            phase="validation"
        )


if __name__ == '__main__':
    logger.info("Starting the MLE training process...")
    main()
    logger.info("MLE training completed.")
