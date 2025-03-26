#!/usr/bin/env python3
"""
main_bootstrap_test.py

This script evaluates four T5 models against three GPT variants (One-Shot Fixed, One-Shot Random, and Zero-Shot)
using BARTScore and ROUGE-L F-score.

It performs the following steps:
  1. Loads test results from CSV files for each GPT variant and each T5 model.
  2. Computes per-sample similarity scores using the "Edited Ending" (predicted scores).
  3. Computes an "edited + Δₘ₁" score for each sample as:
         (Edited + Δₘ₁) = M(Predicted, Edited) + [M(Predicted, Edited) - M(Predicted, Original)]
                      = 2 * (Predicted Score) - (Score with Original Ending)
  4. Saves updated CSV files with the new computed score columns.
  5. Performs a paired bootstrap test (per-story) to compare each T5 model with each GPT variant,
     both for the predicted scores and for the (Edited + Δₘ₁) scores.
  6. Saves a summary CSV file with the average scores and the calculated p-values,
     including a column for the GPT variant.
"""

import os
import pandas as pd
import numpy as np
import random
from src.BARTScore_metric.bart_score import BARTScorer  # Ensure accessible in PYTHONPATH
from rouge import Rouge  # For ROUGE evaluation

# ------------------------------
# Functions to compute per-sample scores
# ------------------------------

def compute_bart_scores(generated_texts, reference_texts, scorer):
    """
    Computes the BARTScore for each generated text compared to its reference.
    
    Args:
        generated_texts (list of str): Generated texts (predictions).
        reference_texts (list of str): Reference texts (e.g., Edited Ending).
        scorer (BARTScorer): Instance of BARTScorer.
        
    Returns:
        list of float: BART scores for each sample.
    """
    scores = []
    for gen, ref in zip(generated_texts, reference_texts):
        try:
            # scorer.score() expects lists as input.
            score_list = scorer.score([gen], [ref], batch_size=1)
            scores.append(score_list[0])
        except Exception as e:
            print(f"Error computing BART score: {e}")
            scores.append(float('nan'))
    return scores

def compute_rouge_scores(generated_texts, reference_texts, rouge_evaluator):
    """
    Computes the ROUGE-L F-score for each generated text compared to its reference.
    
    Args:
        generated_texts (list of str): Generated texts.
        reference_texts (list of str): Corresponding reference texts.
        rouge_evaluator (Rouge): Instance of the Rouge evaluator.
        
    Returns:
        list of float: ROUGE-L F-scores for each sample.
    """
    scores = []
    for gen, ref in zip(generated_texts, reference_texts):
        try:
            score_dict = rouge_evaluator.get_scores(gen, ref)[0]
            scores.append(score_dict['rouge-l']['f'])
        except Exception as e:
            print(f"Error computing ROUGE score: {e}")
            scores.append(float('nan'))
    return scores

# ------------------------------
# Paired Bootstrap Test Function with Enhanced Prints
# ------------------------------

def bootstrap_test_for_systems(scores_A, scores_B, num_samples=12120, metric_label=""):
    """
    Performs a one-sided bootstrap test on paired differences computed from two lists of scores.
    It computes the paired differences as: differences = scores_A - scores_B.
    
    The null hypothesis is that the mean difference is zero.
    
    Process:
      1. Compute the paired differences.
      2. Calculate the observed mean difference.
      3. Resample these differences with replacement for num_samples iterations.
      4. For each bootstrap sample, compute its mean.
      5. The p-value is the proportion of bootstrap sample means that are >= the observed mean difference.
    
    Args:
        scores_A (list of float): Scores for system A (e.g., T5 predicted scores).
        scores_B (list of float): Scores for system B (e.g., GPT predicted scores).
        num_samples (int): Number of bootstrap iterations.
        metric_label (str): Label for the metric (e.g., "BART" or "ROUGE").
    
    Returns:
        float: The empirical one-sided p-value.
    """
    differences = np.array(scores_A) - np.array(scores_B)
    observed_diff = np.mean(differences)
    if metric_label:
        print(f"[{metric_label} Bootstrap] Observed mean difference: {observed_diff:.12e}")
    else:
        print(f"[Bootstrap] Observed mean difference: {observed_diff:.12e}")
    
    n = len(differences)
    r = 0  # Count how many bootstrap samples have a mean >= observed_diff.
    for i in range(num_samples):
        sample_diff = np.random.choice(differences, size=n, replace=True)
        sample_mean = np.mean(sample_diff)
        if i < 5:  # Print first 5 iterations.
            if metric_label:
                print(f"[{metric_label} Bootstrap] Iteration {i+1}: Sample mean = {sample_mean:.12e}")
            else:
                print(f"[Bootstrap] Iteration {i+1}: Sample mean = {sample_mean:.12e}")
        if sample_mean >= 2*observed_diff:
            r += 1
    p_value = float(r) / num_samples
    if metric_label:
        print(f"[{metric_label} Bootstrap] {r} out of {num_samples} iterations met the condition.")
        print(f"[{metric_label} Bootstrap] Calculated p-value: {p_value:.12e}")
    else:
        print(f"[Bootstrap] {r} out of {num_samples} iterations met the condition.")
        print(f"[Bootstrap] Calculated p-value: {p_value:.12e}")
    return p_value

# ------------------------------
# Function to load a CSV file into a DataFrame
# ------------------------------

def load_dataframe(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path)

# ------------------------------
# Main function: Processing and Evaluation
# ------------------------------

def main():
    # Define input file paths for T5 models.
    t5_files = {
        "T5-Base 5-1": "/data/agirard/Projects/Timetravel/models/model_2024-09-03-17/test_details.csv",
        "T5-Base 10-1": "/data/agirard/Projects/Timetravel/models/model_2024-09-03-20/test_details.csv",
        "T5-Large 5-1": "/data/agirard/Projects/Timetravel/models/model_2024-08-30-11/test_details.csv",
        "T5-Large 10-1": "/data/agirard/Projects/Timetravel/models/model_2024-08-30-06/test_details.csv"
    }
    # Define GPT variants with their corresponding file paths.
    gpt_files = {
        "GPT_OneShotFixed": "/data/agirard/Projects/Timetravel/results/one_shot_results_fixed.csv",
        "GPT_OneShotRandom": "/data/agirard/Projects/Timetravel/results/one_shot_results_random.csv",
        "GPT_ZeroShot": "/data/agirard/Projects/Timetravel/results/zero_shot_results.csv"
    }

    # Define output directory.
    output_dir = "/home/agirard/Data/Projects/Timetravel/results/Bootstrap"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize evaluators.
    bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
    rouge_evaluator = Rouge()

    # We'll create a list to store summary results for each combination.
    summary_results = []

    # Loop over each GPT variant.
    for gpt_label, gpt_file_path in gpt_files.items():
        print(f"\n=== Processing {gpt_label} Results ===")
        gpt_df = load_dataframe(gpt_file_path)
        gpt_generated = gpt_df['generated_text'].tolist()
        gpt_edited = gpt_df['edited_ending'].tolist()
        gpt_original = gpt_df['original_ending'].tolist()

        # Compute GPT scores for this variant.
        gpt_bart_predicted = compute_bart_scores(gpt_generated, gpt_edited, bart_scorer)
        gpt_rouge_predicted = compute_rouge_scores(gpt_generated, gpt_edited, rouge_evaluator)
        gpt_bart_original = compute_bart_scores(gpt_generated, gpt_original, bart_scorer)
        gpt_rouge_original = compute_rouge_scores(gpt_generated, gpt_original, rouge_evaluator)
        # Compute "edited + Δ_M₁" scores.
        gpt_bart_plusDelta = [2 * p - o for p, o in zip(gpt_bart_predicted, gpt_bart_original)]
        gpt_rouge_plusDelta = [2 * p - o for p, o in zip(gpt_rouge_predicted, gpt_rouge_original)]

        # Add computed columns to GPT DataFrame.
        gpt_df['BART_Score_Predicted'] = gpt_bart_predicted
        gpt_df['BART_Score_plusDelta'] = gpt_bart_plusDelta
        gpt_df['ROUGE_Score_Predicted'] = gpt_rouge_predicted
        gpt_df['ROUGE_Score_plusDelta'] = gpt_rouge_plusDelta

        # Save updated GPT file with variant label.
        gpt_output_path = os.path.join(output_dir, f"{gpt_label}_withScores.csv")
        gpt_df.to_csv(gpt_output_path, index=False)
        print(f"Saved updated {gpt_label} file to {gpt_output_path}")

        # Compute average scores for GPT for this variant.
        gpt_avg_bart_pred = np.nanmean(gpt_bart_predicted)
        gpt_avg_bart_plus = np.nanmean(gpt_bart_plusDelta)
        gpt_avg_rouge_pred = np.nanmean(gpt_rouge_predicted)
        gpt_avg_rouge_plus = np.nanmean(gpt_rouge_plusDelta)

        # Loop over each T5 model.
        print("\n=== Processing T5 Models for Comparison with " + gpt_label + " ===")
        for model_name, file_path in t5_files.items():
            print(f"\n--- Processing Model: {model_name} ---")
            t5_df = load_dataframe(file_path)
            t5_generated = t5_df['Generated Text'].tolist()
            t5_edited = t5_df['Edited Ending'].tolist()
            t5_original = t5_df['Original Ending'].tolist()

            # Compute T5 predicted and original scores.
            t5_bart_predicted = compute_bart_scores(t5_generated, t5_edited, bart_scorer)
            t5_rouge_predicted = compute_rouge_scores(t5_generated, t5_edited, rouge_evaluator)
            t5_bart_original = compute_bart_scores(t5_generated, t5_original, bart_scorer)
            t5_rouge_original = compute_rouge_scores(t5_generated, t5_original, rouge_evaluator)
            # Compute T5 "edited + Δ_M₁" scores.
            t5_bart_plusDelta = [2 * p - o for p, o in zip(t5_bart_predicted, t5_bart_original)]
            t5_rouge_plusDelta = [2 * p - o for p, o in zip(t5_rouge_predicted, t5_rouge_original)]

            # Add new score columns to the T5 DataFrame.
            t5_df['BART_Score_Predicted'] = t5_bart_predicted
            t5_df['BART_Score_plusDelta'] = t5_bart_plusDelta
            t5_df['ROUGE_Score_Predicted'] = t5_rouge_predicted
            t5_df['ROUGE_Score_plusDelta'] = t5_rouge_plusDelta

            t5_output_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_withScores.csv")
            t5_df.to_csv(t5_output_path, index=False)
            print(f"Saved updated file for {model_name} to {t5_output_path}")

            # Calculate paired bootstrap p-values for predicted scores.
            print(f"\nCalculating paired bootstrap p-values for {model_name} (Predicted Scores) comparing to {gpt_label}...")
            p_value_bart_pred = bootstrap_test_for_systems(t5_bart_predicted, gpt_bart_predicted, num_samples=12120, metric_label="BART (Predicted)")
            p_value_rouge_pred = bootstrap_test_for_systems(t5_rouge_predicted, gpt_rouge_predicted, num_samples=12120, metric_label="ROUGE (Predicted)")
            print(f"{model_name} (Predicted Scores) vs {gpt_label}:")
            print(f"  BART p-value = {p_value_bart_pred:.12e}")
            print(f"  ROUGE p-value = {p_value_rouge_pred:.12e}")

            # Calculate paired bootstrap p-values for "edited + Δ_M₁" scores.
            print(f"\nCalculating paired bootstrap p-values for {model_name} (Edited + Δ_M₁ Scores) comparing to {gpt_label}...")
            p_value_bart_plus = bootstrap_test_for_systems(t5_bart_plusDelta, gpt_bart_plusDelta, num_samples=12120, metric_label="BART (Edited + Δ_M₁)")
            p_value_rouge_plus = bootstrap_test_for_systems(t5_rouge_plusDelta, gpt_rouge_plusDelta, num_samples=12120, metric_label="ROUGE (Edited + Δ_M₁)")
            print(f"{model_name} (Edited + Δ_M₁ Scores) vs {gpt_label}:")
            print(f"  BART p-value = {p_value_bart_plus:.12e}")
            print(f"  ROUGE p-value = {p_value_rouge_plus:.12e}")

            # Append summary results for this model and GPT variant.
            summary_results.append({
                "GPT_Variant": gpt_label,
                "Model": model_name,
                "T5_Avg_BART_Predicted": np.nanmean(t5_bart_predicted),
                "GPT_Avg_BART_Predicted": gpt_avg_bart_pred,
                "Bootstrap_p_BART_Predicted": p_value_bart_pred,
                "T5_Avg_BART_plusDelta": np.nanmean(t5_bart_plusDelta),
                "GPT_Avg_BART_plusDelta": gpt_avg_bart_plus,
                "Bootstrap_p_BART_plusDelta": p_value_bart_plus,
                "T5_Avg_ROUGE_Predicted": np.nanmean(t5_rouge_predicted),
                "GPT_Avg_ROUGE_Predicted": gpt_avg_rouge_pred,
                "Bootstrap_p_ROUGE_Predicted": p_value_rouge_pred,
                "T5_Avg_ROUGE_plusDelta": np.nanmean(t5_rouge_plusDelta),
                "GPT_Avg_ROUGE_plusDelta": gpt_avg_rouge_plus,
                "Bootstrap_p_ROUGE_plusDelta": p_value_rouge_plus
            })

    # Save the summary results to a CSV file.
    summary_df = pd.DataFrame(summary_results)
    summary_output_path = os.path.join(output_dir, "Bootstrap_Comparison_Summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"\nSaved bootstrap summary results to {summary_output_path}")

if __name__ == "__main__":
    main()
