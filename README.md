# TimeTravel-PolicyGradientRL: Reinforcement Learning with Counterfactual Metrics as Reward for Story Rewriting

This project investigates the use of **Reinforcement Learning (RL)**, particularly policy gradient methods, to improve model performance in **counterfactual story rewriting**. By treating text generation as a sequential decision-making problem, we train models explicitly using counterfactual evaluation metrics such as **BLEU**, **ROUGE**, **BERTScore**, and **BARTScore** as reward functions, with a regularization term subtracted to prevent overfitting.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Reward Function and Regularization](#reward-function-and-regularization)
5. [Usage](#usage)
   - [Installation](#installation)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
6. [License](#license)

---

## Overview

In counterfactual story rewriting, models modify a story’s ending to reflect a hypothetical event, requiring minimal yet accurate changes while maintaining narrative coherence. 

This project applies **policy gradient** RL methods to improve model performance by directly optimizing for **counterfactual rewriting metrics** such as BLEU, ROUGE, BERTScore, and BARTScore. We use the scores of these metrics as reward functions in our experiments, while subtracting a regularization term to control overfitting and ensure minimal necessary changes.

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

In this project, we explore the use of **BLEU**, **ROUGE**, **BERTScore**, and **BARTScore** as reward functions for training our model using policy gradient methods. The model's goal is to generate coherent counterfactual story endings that maximize the score of the chosen metric.

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

This allows the model to focus on generating high-quality story endings that score well according to the selected metric, while the regularization term ensures the model doesn't make unnecessary large changes.

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

You can train the model using policy gradient methods and select a reward function (BLEU, ROUGE, BERTScore, or BARTScore):

```bash
python src/main_t5.py --reward_metric BLEU
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
```

