# src/utils/metrics.py

import logging
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer
from src.BARTScore_metric.bart_score import BARTScorer
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    A class for evaluating text generation models using various metrics such as BLEU, ROUGE, BERTScore, and BARTScore.
    """

    def __init__(self):
        """
        Initializes the metric evaluators with configurations from the CONFIG file.
        """
        self.sacre_bleu = BLEU()  # SacreBLEU score evaluator
        self.rouge = Rouge()  # ROUGE score evaluator
        self.bert_scorer = BERTScorer(
            model_type=CONFIG["bert_scorer_model_type"],
            device=CONFIG["scorer_device"],
            num_layers=None,
            batch_size=CONFIG["bert_scorer_batch_size"]
        )  # BERTScore evaluator
        self.bart_scorer = BARTScorer(
            device=CONFIG["scorer_device"],
            checkpoint=CONFIG["bart_scorer_checkpoint"]
        )  # BARTScore evaluator

    def calculate_and_log_bleu_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs SacreBLEU scores for various comparisons of generated texts and references.
        """
        # Prepare references for BLEU score calculation
        edited_endings_refs = [[ending] for ending in all_edited_endings] if all_edited_endings else None
        counterfactuals_refs = [[cf] for cf in all_counterfactuals]
        initials_refs = [[init] for init in all_initials]
        original_endings_refs = [[orig] for orig in all_original_endings]

        # Define all comparisons to calculate BLEU scores
        all_comparisons = [
            ('bleu_prediction_edited', all_generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', all_generated_texts, counterfactuals_refs),
            ('bleu_prediction_initial', all_generated_texts, initials_refs),
            ('bleu_prediction_original', all_generated_texts, original_endings_refs),
            ('bleu_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bleu_edited_ending_initial', all_edited_endings, all_initials),
            ('bleu_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Calculate and log BLEU scores for each comparison
        bleu_scores = {}
        for label, texts, references in all_comparisons:
            if references is not None:
                try:
                    bleu_result = self.sacre_bleu.corpus_score(texts, references)
                    bleu_score = bleu_result.score
                    logger.info(f"{label}: {bleu_score}")
                    bleu_scores[label] = bleu_score
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bleu_scores[label] = 'N/A'
        return bleu_scores

    def calculate_and_log_rouge_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs ROUGE scores for various comparisons of generated texts and references.
        """
        # Define all comparisons to calculate ROUGE scores
        all_comparisons = [
            ('rouge_prediction_edited', all_generated_texts, all_edited_endings),
            ('rouge_prediction_cf', all_generated_texts, all_counterfactuals),
            ('rouge_prediction_initial', all_generated_texts, all_initials),
            ('rouge_prediction_original', all_generated_texts, all_original_endings),
            ('rouge_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('rouge_edited_ending_initial', all_edited_endings, all_initials),
            ('rouge_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Calculate and log ROUGE scores for each comparison
        rouge_scores = {}
        for label, hypotheses, references in all_comparisons:
            if references:
                try:
                    rouge_scores_set = self.rouge.get_scores(hypotheses, references, avg=True)
                    for score_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                        rouge_scores[f"{label}_{score_type}_f"] = rouge_scores_set[score_type]['f']
                        rouge_scores[f"{label}_{score_type}_p"] = rouge_scores_set[score_type]['p']
                        rouge_scores[f"{label}_{score_type}_r"] = rouge_scores_set[score_type]['r']
                        logger.info(f"{label}_{score_type}_f: {rouge_scores_set[score_type]['f']}")
                        logger.info(f"{label}_{score_type}_p: {rouge_scores_set[score_type]['p']}")
                        logger.info(f"{label}_{score_type}_r: {rouge_scores_set[score_type]['r']}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    rouge_scores[f"{label}_f"] = 'N/A'
                    rouge_scores[f"{label}_p"] = 'N/A'
                    rouge_scores[f"{label}_r"] = 'N/A'
        return rouge_scores

    def calculate_and_log_bert_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs BERT similarity scores for various comparisons of generated texts and references.
        """
        # Define all comparisons to calculate BERT similarity scores
        all_comparisons = [
            ('bert_prediction_edited', all_generated_texts, all_edited_endings),
            ('bert_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bert_prediction_initial', all_generated_texts, all_initials),
            ('bert_prediction_original', all_generated_texts, all_original_endings),
            ('bert_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bert_edited_ending_initial', all_edited_endings, all_initials),
            ('bert_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Calculate and log BERT similarity scores for each comparison
        bert_scores = {}
        for label, texts_a, texts_b in all_comparisons:
            if texts_b:
                try:
                    P, R, F1 = self.bert_scorer.score(texts_a, texts_b)
                    avg_precision = P.mean().item()
                    avg_recall = R.mean().item()
                    avg_f1 = F1.mean().item()
                    logger.info(f"{label}_precision: {avg_precision}")
                    logger.info(f"{label}_recall: {avg_recall}")
                    logger.info(f"{label}_f1: {avg_f1}")
                    bert_scores[f"{label}_precision"] = avg_precision
                    bert_scores[f"{label}_recall"] = avg_recall
                    bert_scores[f"{label}_f1"] = avg_f1
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bert_scores[f"{label}_precision"] = 'N/A'
                    bert_scores[f"{label}_recall"] = 'N/A'
                    bert_scores[f"{label}_f1"] = 'N/A'
        return bert_scores

    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs BART-based similarity scores for a variety of text comparisons,
        using the BARTScorer to evaluate the similarity between different segments of texts.
        """
        # Define all pairs of text segments for which to calculate similarity scores
        all_comparisons = [
            ('bart_prediction_edited', all_generated_texts, all_edited_endings),
            ('bart_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bart_prediction_initial', all_generated_texts, all_initials),
            ('bart_prediction_original', all_generated_texts, all_original_endings),
            ('bart_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bart_edited_ending_initial', all_edited_endings, all_initials),
            ('bart_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        # Calculate and log BARTScores for each comparison
        bart_scores = {}
        for label, src_texts, tgt_texts in all_comparisons:
            if tgt_texts:
                try:
                    if isinstance(tgt_texts[0], list):
                        scores = self.bart_scorer.multi_ref_score(src_texts, tgt_texts, agg='mean', batch_size=4)
                    else:
                        scores = self.bart_scorer.score(src_texts, tgt_texts, batch_size=4)
                    avg_score = sum(scores) / len(scores) if scores else float('nan')
                    logger.info(f"{label}_avg_score: {avg_score}")
                    bart_scores[f"{label}_avg_score"] = avg_score
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bart_scores[f"{label}_avg_score"] = 'N/A'
        return bart_scores
