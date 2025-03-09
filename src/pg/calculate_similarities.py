import os
import pandas as pd
import logging
from src.pg.utils.metrics import MetricsEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data(df):
    generated_texts = df['Generated Text'].tolist()
    edited_endings = df['Edited Ending'].tolist()
    counterfactuals = df['Counterfactual'].tolist()
    initials = df['Initial'].tolist()
    premises = df['Premise'].tolist()
    original_endings = df['Original Ending'].tolist()

    evaluator = MetricsEvaluator()
    all_metrics = {}

    all_metrics.update(evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    all_metrics.update(evaluator.calculate_and_log_bert_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    all_metrics.update(evaluator.calculate_and_log_bleu_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    all_metrics.update(evaluator.calculate_and_log_rouge_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']

    return metrics_df


def process_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        metrics_df = process_data(df)

        base_dir = os.path.dirname(file_path)
        base_name = os.path.basename(file_path).replace('.csv', '')
        output_file_path = os.path.join(base_dir, f'{base_name}_metrics.csv')

        metrics_df.to_csv(output_file_path, index=False)
        print(f"Metrics saved to {output_file_path}")
    else:
        print(f"File not found: {file_path}")


def main():
    file_path = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-03-04-07/test_details_pg_20250308_023111.csv'
    process_file(file_path)


if __name__ == "__main__":
    main()
