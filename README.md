# TimeTravel-PolicyGradientRL: Reinforcement Learning with Metrics-Based Reward Functions for Counterfactual Story Rewriting

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Reward and Regularization Formula](#reward-and-regularization-formula)
5. [Usage](#usage)
   - [Installation](#installation)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
6. [License](#license)

---

## Overview

This project enhances counterfactual story rewriting using reinforcement learning (RL), with a focus on using text similarity metrics (BLEU, ROUGE, BERTScore, and BARTScore) as reward functions. Additionally, we include regularization to prevent overfitting and ensure balanced learning. The model adjusts a story’s ending in response to counterfactual events, making minimal but precise changes while maintaining narrative coherence.

---

## Key Features

- **Reinforcement Learning with Regularized Custom Rewards**: The model optimizes based on custom reward functions derived from text similarity metrics and includes a regularization term (\(\lambda = 0.5\)) to stabilize learning and prevent overfitting.
  
- **Custom Training Objective**: A training objective that integrates rewards from multiple evaluation metrics, enhancing model performance on tasks requiring minimal intervention and narrative coherence.

- **Metrics Integration**: Incorporates BLEU, ROUGE, BERTScore, and BARTScore during training to guide the model towards generating more accurate and contextually appropriate counterfactual story endings.

---

## Project Structure

```bash
TimeTravel-PolicyGradientRL/
├── src/
│   ├── models/
│   │   └── model_T5.py             # The T5 fine-tuner model with custom loss, RL, and regularization.
│   ├── utils/
│   │   ├── config.py               # Configuration file for paths, parameters, and reward-related settings.
│   │   ├── metrics.py              # Custom evaluation metrics used as rewards.
│   │   └── utils.py                # Utility functions (data preprocessing, differential weights).
│   ├── data_loader.py              # DataLoader for processing and batching JSON data.
│   ├── main_t5.py                  # Main training script with RL and regularization integration.
│   └── main_t5_metrics.py          # Evaluation script to calculate and log post-training metrics.
├── results/                        # Directory to store training logs, evaluation metrics, and checkpoints.
├── data/                           # Directory to store datasets and processed data.
└── README.md                       # This README file.
```

---

## Reward and Regularization Formula

The reward \( R_{\text{total}} \) is calculated as a weighted sum of BLEU, ROUGE, BERTScore, and BARTScore, with a regularization term \( \lambda = 0.5 \) to prevent overfitting.

\[
R_{\text{total}} = w_{\text{BLEU}} R_{\text{BLEU}} + w_{\text{ROUGE}} R_{\text{ROUGE}} + w_{\text{BERT}} R_{\text{BERT}} + w_{\text{BART}} R_{\text{BART}} - \lambda
\]

Where \( w_{\text{BLEU}} = w_{\text{ROUGE}} = w_{\text{BERT}} = w_{\text{BART}} = 0.25 \) are the default weights for each metric. The regularization term \( \lambda = 0.5 \) penalizes the reward to prevent the model from overfitting and to promote generalization.

The new loss function, integrating the reward, is:

\[
\mathcal{L}_{\text{new}} = \mathcal{L} - R_{\text{total}}
\]

---

## Usage

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/amelie-girard7/TimeTravel-PolicyGradientRL.git
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

You can train the model using reinforcement learning with metrics as reward functions and regularization by running the following command:

```bash
python src/main_t5.py
```

This will:
- Fine-tune the T5 model on the counterfactual rewriting task.
- Use BLEU, ROUGE, BERTScore, and BARTScore as reward functions in the training loop, with regularization.
- Save model checkpoints and log the training progress in the `results/` directory.

### Evaluating the Model

After training, you can evaluate the model's performance using the following command:

```bash
python src/main_t5_metrics.py --model_checkpoint /path/to/checkpoint.ckpt
```

This will calculate BLEU, ROUGE, BERTScore, and BARTScore on the validation or test set and save the results to `results/metrics`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
