# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/pg/main_pg.py

import sys
import os
import datetime
import logging
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # Added EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.pg.models.model import FlanT5FineTuner
from src.pg.data_loader import create_dataloaders
from src.pg.utils.metrics import MetricsEvaluator
from src.pg.utils.config import CONFIG
import pandas as pd
import re

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize or load the model from a checkpoint
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
        logger.info(f"Initializing fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(CONFIG["model_name"], model_dir, file_label=file_label)

    return model


# Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing
def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger):
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        # callbacks=[checkpoint_callback],
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=0.1,
        default_root_dir="./"
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer


# Evaluate and save metrics after testing/validation
# def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
#     logger.info(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")
#     model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)
#     trainer = Trainer(accelerator='gpu', devices=1)

#     # Running evaluation
#     if phase == "test":
#         trainer.test(model, loader, verbose=False)
#     elif phase == "validation":
#         trainer.validate(model, loader, verbose=False)
#     else:
#         raise ValueError(f"Unknown phase: {phase}")

#     details_file = os.path.join(model_dir, f"{phase}_details{file_label}.csv")
#     if not os.path.exists(details_file):
#         logger.error(f"{phase.capitalize()} details file not found at {details_file}")
#         raise FileNotFoundError(f"{phase.capitalize()} details file not found at {details_file}")

#     details_df = pd.read_csv(details_file)
#     filtered_details = details_df[details_df['Epoch'] == best_epoch]
#     if filtered_details.empty:
#         logger.warning(f"No rows found for epoch {best_epoch} in {phase}_details_df. Evaluating all rows instead.")
#         filtered_details = details_df

#     generated_texts = filtered_details['Generated Text'].tolist()
#     edited_endings = filtered_details['Edited Ending'].tolist()
#     counterfactuals = filtered_details['Counterfactual'].tolist()
#     initials = filtered_details['Initial'].tolist()
#     premises = filtered_details['Premise'].tolist()
#     original_endings = filtered_details['Original Ending'].tolist()

#     if not (generated_texts and edited_endings):
#         logger.error(f"Generated texts or edited endings are empty. Skipping metric calculations for {phase}.")
#         return

#     evaluator = MetricsEvaluator()
#     metrics = {}

#     try:
#         metrics.update(evaluator.calculate_and_log_bart_similarity(generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger))
#         metrics.update(evaluator.calculate_and_log_bert_similarity(generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger))
#         metrics.update(evaluator.calculate_and_log_bleu_scores(generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger))
#         metrics.update(evaluator.calculate_and_log_rouge_scores(generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger))
#     except Exception as e:
#         logger.error(f"Error calculating metrics for {phase}: {e}")

#     metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
#     metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
#     metrics_df.reset_index(inplace=True)
#     metrics_df.columns = ['Metric', 'Score']
#     metrics_df.to_csv(metrics_file, index=False)

#     logger.info(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")


# Extract the epoch number from the checkpoint filename
def extract_epoch_from_checkpoint(checkpoint_path):
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    logger.warning(f"Could not extract epoch from checkpoint path: {checkpoint_path}")
    return "Unknown"


# Main execution logic
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"pg_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False
    )
    wandb_logger.experiment.config.update(CONFIG)

    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )

    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], CONFIG["test_file"].split('.')[0]

    # PG Phase
    model = setup_model(
        model_dir,
        file_label="_pg",
        checkpoint_path=CONFIG["pg_checkpoint_path"]
    )

    pg_checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        monitor='validation_pg_loss',
        mode='min',
        save_top_k=1,
        filename="pg_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss-{validation_pg_loss:.2f}"
    )

    # Early stopping callback to stop training when the validation loss stops improving
    early_stop_callback = EarlyStopping(
        monitor='validation_pg_loss',  # Monitor the validation loss metric
        min_delta=0.00,                # Minimum change in the monitored metric to qualify as an improvement
        patience=2,                    # Number of epochs to wait without improvement before stopping training
        verbose=True,                  # Print messages when early stopping is triggered
        mode='min'                     # We expect the monitored metric to decrease; training stops when it stops decreasing
    )   


    # trainer = setup_trainer(CONFIG["pg_epochs"], pg_checkpoint_callback, wandb_logger)
    trainer = setup_trainer(CONFIG["pg_epochs"], pg_checkpoint_callback, early_stop_callback, wandb_logger)
 
    trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])

    best_checkpoint = pg_checkpoint_callback.best_model_path
    best_epoch = extract_epoch_from_checkpoint(best_checkpoint)

    # Load explicitly the best checkpoint
    model = setup_model(model_dir, file_label="_pg", checkpoint_path=best_checkpoint)

    # Explicitly set up Trainer without logging for final evaluation
    trainer = Trainer(accelerator='gpu', devices=1, logger=False)


    # Run explicit validation pass to collect and log details
    trainer.validate(model, dataloaders[dev_key], verbose=False)


    # evaluate_and_save(
    #     model_dir=model_dir,
    #     loader=dataloaders[test_key],
    #     best_checkpoint=best_checkpoint,
    #     file_label="_pg",
    #     best_epoch=best_epoch,
    #     phase="test"
    # )
    # evaluate_and_save(
    #     model_dir=model_dir,
    #     loader=dataloaders[dev_key],
    #     best_checkpoint=best_checkpoint,
    #     file_label="_pg",
    #     best_epoch=best_epoch,
    #     phase="validation"
    # )


if __name__ == '__main__':
    logger.info("Starting the PG process...")
    main()
    logger.info("Process completed.")
