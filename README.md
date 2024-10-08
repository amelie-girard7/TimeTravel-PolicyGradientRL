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
7. [Usage](#usage)
   - [Installation](#installation)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
8. [License](#license)

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
  - Introduces a regularization term $ \lambda = 0.5 $ to prevent overfitting and ensure a balanced learning process.

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

The reward $ R(y) $ for a generated ending $ y $ is calculated as:

\[
R(y) = \text{MetricScore}(y) - 0.5
\]

Where:
- **MetricScore** is the score calculated by the chosen metric (BLEU, ROUGE, BERTScore, or BARTScore) for the generated ending compared to the reference (true) ending.
- $ 0.5 $ is a regularization term that penalizes excessive changes and helps prevent overfitting.

### Loss Function with Reward

The policy gradient method optimizes the model using the reward function defined above. The loss function for training becomes:

\[
L_{\text{new}} = L_{\text{MLE}} - \mathbb{E}[R(y)]
\]

Where:
- $ L_{\text{MLE}} $ is the standard Maximum Likelihood Estimation loss.
- $ R(y) $ is the reward derived from the chosen metric (BLEU, ROUGE, BERTScore, or BARTScore).

---

## Second Step: Delta-Based Reward

In the second phase of our training, we incorporate a delta-based reward function. Here, instead of directly using the raw metric scores, we use the **difference** (delta) between the generated ending and the reference, aiming to penalize deviations away from zero, whether positive or negative.

The delta-based reward $ R_{\text{delta}}(y) $ is computed as:

\[
R_{\text{delta}}(y) = \Delta_{M_1}(y) - \lambda \cdot | \Delta_{M_2}(y) |
\]

Where:
- $ \Delta_{M_1}(y) $ measures how much the generated ending aligns with the edited ending compared to the original ending.
- $ \Delta_{M_2}(y) $ penalizes any unnecessary deviation from the counterfactual event, ensuring minimal yet appropriate changes.
- $ \lambda = 0.5 $ controls the trade-off between alignment and minimal intervention.

This approach ensures that the model is not only rewarded for matching the reference ending but also penalized for making excessive or unnecessary changes.

---

## Task Definition and Notation

In this section, we define the counterfactual story rewriting task and introduce the notation used in this project.

### Task Definition and Components

Each story has the following components:

- **Premise** ($ X_P $): The foundational scenario or context for the story.
- **Initial Event** ($ X_{IE} $): An event that leads to the story’s original conclusion.
- **Original Ending** ($ X_{OE} $): The story’s original conclusion.
- **Counterfactual Event** ($ X_{CE} $): A hypothetical event that contradicts the initial event and changes the story.
- **Edited Ending** ($ Y_{EE} $): The modified conclusion reflecting the counterfactual event.

### Task Objective

The task is to generate an **edited ending** $ Y_{EE} $ given the **premise** $ X_P $, **counterfactual event** $ X_{CE} $, and **original ending** $ X_{OE} $, while making minimal yet appropriate changes to $ X_{OE} $ to reflect $ X_{CE} $.

### Key Aspects of the Task

- **Minimal Intervention**: Adjust $ X_{OE} $ with minimal changes to align it with $ X_{CE} $.
- **Narrative Coherence**: Ensure that changes maintain the story’s overall coherence and consistency.
- **Counterfactual Adaptability**: Incorporate the counterfactual event $ X_{CE} $ effectively into the edited ending.

### Example

| **Initial Scenario** | **Counterfactual Scenario** |
|----------------------|-----------------------------|
| **Premise**: John has a headache. | **Premise**: John has a headache. |
| **Initial Event**: He takes aspirin. | **Counterfactual Event**: He takes an experimental pill. |
| **Original Ending**: John waits for hours before feeling better. | **Edited Ending**: John feels better within minutes. |

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

This project is licensed under the MIT License - see

 the [LICENSE](LICENSE) file for details.
