# TimeTravel-PolicyGradientRL: Reinforcement Learning with Counterfactual Metrics as Reward for Story Rewriting

This project investigates the use of **Reinforcement Learning (RL)**, particularly policy gradient methods, to improve model performance in **counterfactual story rewriting**. By treating text generation as a sequential decision-making problem, we train models explicitly using counterfactual evaluation metrics such as **BLEU**, **ROUGE**, **BERTScore**, and **BARTScore** as reward functions, with a regularization term subtracted to prevent overfitting. In a second step, we explore training with the difference (delta) of these metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Reward Function and Regularization](#reward-function-and-regularization)
5. [Second Step: Delta-Based Reward](#second-step-delta-based-reward)
6. [Task Definition and Notation](#task-definition-and-notation)
7. [Solution Implementation](#solution-implementation)
   - [Using Individual Metric Scores](#using-individual-metric-scores)
   - [Combining Metric Scores for Rewards](#combining-metric-scores-for-rewards)
   - [Delta-Based Reward Functions](#delta-based-reward-functions)
8. [Usage](#usage)
   - [Installation](#installation)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
9. [License](#license)

---

## Overview

In counterfactual story rewriting, models modify a story’s ending to reflect a hypothetical event, requiring minimal yet accurate changes while maintaining narrative coherence.

This project applies **policy gradient** RL methods to improve model performance by directly optimizing for **counterfactual rewriting metrics** such as BLEU, ROUGE, BERTScore, and BARTScore. We use the scores of these metrics as reward functions in our experiments, and in a second phase, we explore training based on the difference (delta) of these metrics to fine-tune the model's performance.

---

## Key Features

- **Reinforcement Learning with Counterfactual Metrics as Reward**:

  - Applies policy gradient methods to optimize text generation based on non-differentiable evaluation metrics.
  - Uses BLEU, ROUGE, BERTScore, and BARTScore as rewards to guide the model's behavior.

- **Regularization Term**:

  - Introduces a regularization term \$ \lambda = 0.5 \$ to prevent overfitting and ensure a balanced learning process.

- **Metric Comparison**:

  - Investigates and compares the effectiveness of different metrics as reward functions for RL.

---

## Project Structure

```bash
TimeTravel-PolicyGradientRL/
├── src/
│   ├── models/
│   │   └── model_T5.py             # T5 model with policy gradient implementation.
│   ├── utils/
│   │   ├── config.py               # Configuration for paths, parameters, reward settings.
│   │   ├── metrics.py              # Evaluation metrics (BLEU, ROUGE, BERTScore, BARTScore).
│   │   └── utils.py                # Utility functions (data preprocessing, differential weights).
│   ├── data_loader.py              # DataLoader for JSON data processing.
│   ├── main_t5.py                  # Main script for training with RL and metric rewards.
│   └── main_t5_metrics.py          # Evaluation script for post-training metrics comparison.
├── results/                        # Directory for logs, metrics, and model checkpoints.
├── data/                           # Directory for datasets and processed data.
└── README.md                       # This README file.
```

---

## Reward Function and Regularization

### Policy Gradient Reward Function

In the first phase of this project, we explore the use of **BLEU**, **ROUGE**, **BERTScore**, and **BARTScore** as reward functions for training our model using policy gradient methods. The model's goal is to generate coherent counterfactual story endings that maximize the score of the chosen metric.

The reward \$ R(y) \$ for a generated ending \$ y \$ is calculated as:

Where:

- **MetricScore** is the score calculated by the chosen metric (BLEU, ROUGE, BERTScore, or BARTScore) for the generated ending compared to the reference (true) ending.
- \$ 0.5 \$ is a regularization term that penalizes excessive changes and helps prevent overfitting.

### Loss Function with Reward

The policy gradient method optimizes the model using the reward function defined above. The loss function for training becomes:

Where:

- \$ L\_{\text{MLE}} \$ is the standard Maximum Likelihood Estimation loss.
- \$ R(y) \$ is the reward derived from the chosen metric (BLEU, ROUGE, BERTScore, or BARTScore).

---

## Second Step: Delta-Based Reward

In the second phase of our training, we incorporate a delta-based reward function. Here, instead of directly using the raw metric scores, we use the **difference** (delta) between the generated ending and the reference, aiming to penalize deviations away from zero, whether positive or negative.

The delta-based reward \$ R\_{\text{delta}}(y) \$ is computed as:

Where:

- \$ \Delta\_{M\_1}(y) \$ measures how much the generated ending aligns with the edited ending compared to the original ending.
- \$ \Delta\_{M\_2}(y) \$ penalizes any unnecessary deviation from the counterfactual event, ensuring minimal yet appropriate changes.
- \$ \lambda = 0.5 \$ controls the trade-off between alignment and minimal intervention.

This approach ensures that the model is not only rewarded for matching the reference ending but also penalized for making excessive or unnecessary changes.

---

## Task Definition and Notation

In this section, we define the counterfactual story rewriting task and introduce the notation used in this project.

### Task Definition and Components

Each story has the following components:

- **Premise** (\$ X\_P \$): The foundational scenario or context for the story.
- **Initial Event** (\$ X\_{IE} \$): An event that leads to the story’s original conclusion.
- **Original Ending** (\$ X\_{OE} \$): The story’s original conclusion.
- **Counterfactual Event** (\$ X\_{CE} \$): A hypothetical event that contradicts the initial event and changes the story.
- **Edited Ending** (\$ Y\_{EE} \$): The modified conclusion reflecting the counterfactual event.

### Task Objective

The task is to generate an **edited ending** \$ Y\_{EE} \$ given the **premise** \$ X\_P \$, **counterfactual event** \$ X\_{CE} \$, and **original ending** \$ X\_{OE} \$, while making minimal yet appropriate changes to \$ X\_{OE} \$ to reflect \$ X\_{CE} \$.

### Key Aspects of the Task

- **Minimal Intervention**: Adjust \$ X\_{OE} \$ with minimal changes to align it with \$ X\_{CE} \$.
- **Narrative Coherence**: Ensure that changes maintain the story’s overall coherence and consistency.
- **Counterfactual Adaptability**: Incorporate the counterfactual event \$ X\_{CE} \$ effectively into the edited ending.

### Example

| **Initial Scenario**                                             | **Counterfactual Scenario**                              |
| ---------------------------------------------------------------- | -------------------------------------------------------- |
| **Premise**: John has a headache.                                | **Premise**: John has a headache.                        |
| **Initial Event**: He takes aspirin.                             | **Counterfactual Event**: He takes an experimental pill. |
| **Original Ending**: John waits for hours before feeling better. | **Edited Ending**: John feels better within minutes.     |

---

## Solution Implementation

In this section, we explain how we implemented the policy gradient reinforcement learning approach using counterfactual metrics as reward functions. The solution is structured into three phases: using individual metric scores, combining multiple metrics, and incorporating delta-based rewards.

### Using Individual Metric Scores

The first step involves using individual metrics (e.g., **BLEU**, **ROUGE**, **BERTScore**, **BARTScore**) to evaluate the quality of the generated endings compared to the reference endings. Each of these metrics is used separately as a reward signal for the policy gradient algorithm.

#### Example Code for Calculating BLEU Score:

```python
from sacrebleu.metrics import BLEU

class MetricsEvaluator:
    def __init__(self):
        self.sacre_bleu = BLEU()

    def calculate_bleu(self, generated_texts, references):
        """
        Calculates BLEU score for generated texts against reference texts.

        Parameters:
            generated_texts (list): List of generated story endings.
            references (list of lists): List containing lists of reference endings.

        Returns:
            bleu_score (float): Computed BLEU score.
        """
        # Calculate BLEU score
        bleu_result = self.sacre_bleu.corpus_score(generated_texts, references)
        bleu_score = bleu_result.score
        return bleu_score
```

The **BLEU** score is used as the reward to guide the model during training. For example, if the generated ending is `"Ryan decided to stay and work on his project"` and the reference ending is `"Ryan decided to focus on work due to his project deadline"`, the calculated BLEU score serves as a reward for that generated sequence.

### Combining Metric Scores for Rewards

After experimenting with individual metrics, we combine multiple metrics to form a comprehensive reward signal. This allows the model to learn from different aspects of the generated text quality, such as fluency (BLEU), relevance (ROUGE), and semantic similarity (BERTScore and BARTScore).

#### Example Code for Combining Rewards:

```python
def calculate_combined_reward(self, generated_text, reference):
    """
    Calculates the reward using different evaluation metrics and combines them.

    Parameters:
        generated_text (str): The generated story ending.
        reference (str): The reference ending.

    Returns:
        final_reward (float): Combined reward value.
    """
    # Calculate BLEU score
    bleu_reward = self.metrics_evaluator.calculate_bleu([generated_text], [[reference]])

    # Calculate ROUGE score
    rouge_reward = self.metrics_evaluator.calculate_rouge([generated_text], [reference])

    # Combine rewards (can be weighted)
    combined_reward = bleu_reward * 0.3 + rouge_reward * 0.7

    # Apply regularization term
    final_reward = combined_reward - 0.5

    return final_reward
```

This combined reward provides a balanced evaluation that leverages different qualities of the generated text. For example, combining **BLEU** and **ROUGE** ensures that the generated text is both fluent and relevant to the reference.

### Delta-Based Reward Functions

In the final phase, we introduce **Delta-Based Reward Functions** to refine the model’s performance further. Here, instead of using the raw metric scores, we use the difference between metrics to ensure that generated endings are not only correct but also minimally different from the original, maintaining coherence and making essential modifications.

#### Delta-Based Reward Calculation:

- **Delta M1 (\$\Delta\_{M\_1}\$)**: Measures how much the generated ending aligns with the reference edited ending compared to the original ending.
- **Delta M2 (\$\Delta\_{M\_2}\$)**: Penalizes unnecessary deviation from the counterfactual, encouraging minimal and necessary changes.

#### Example Code for Delta-Based Reward Calculation:

```python
def calculate_delta_reward(self, generated_text, original_ending, reference):
    """
    Calculate the delta-based reward to balance minimal intervention and alignment.

    Parameters:
        generated_text (str): The generated story ending.
        original_ending (str): The original story ending before edits.
        reference (str): The reference edited ending.

    Returns:
        delta_reward (float): Delta-based reward value.
    """
    # Calculate alignment metric (Delta M1)
    delta_m1 = self.metrics_evaluator.calculate_bleu([generated_text], [[reference]])

    # Calculate minimal change metric (Delta M2)
    delta_m2 = self.metrics_evaluator.calculate_bleu([generated_text], [[original_ending]])

    # Delta-based reward
    delta_reward = delta_m1 - 0.5 * abs(delta_m2)

    return delta_reward
```

This approach ensures that the model is not only rewarded for generating text that matches the reference but also for making the fewest necessary changes. For instance, if the original ending is `"John waited for hours to feel better"` and the reference is `"John felt better within minutes"`, the **Delta M2** penalizes unnecessary deviations from the original story structure while ensuring the necessary changes are made.

---

## Usage

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/amelie-girard7/TimeTravel-PolicyGradientRL.git
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

You can train the model using policy gradient methods with a selected reward function (BLEU, ROUGE, BERTScore, or BARTScore):

```bash
python src/main_t5.py --reward_metric BLEU
```

In the second phase, you can train using the delta-based reward function:

```bash
python src/main_t5.py --reward_metric DELTA
```

The script will:

- Fine-tune the T5 model using RL with the selected reward metric.
- Save model checkpoints and logs in the `results/` directory.

### Evaluating the Model

After training, evaluate the model using:

```bash
python src/main_t5_metrics.py --model_checkpoint /path/to/checkpoint.ckpt
```

This will compute the metrics (BLEU, ROUGE, BERTScore, BARTScore) for comparison and save the results in `results/metrics`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
