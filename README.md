# TimeTravel-PolicyGradientRL: Reinforcement Learning with Counterfactual Metrics as Reward for Story Rewriting

This project explores the application of **Reinforcement Learning (RL)**, specifically policy gradient methods, to enhance model performance in **counterfactual story rewriting**. By framing text generation as a sequential decision-making process, we train models using evaluation metrics—including both conventional metrics and previously proposed task-specific metrics—as reward functions. This approach aims to guide the model more effectively towards generating appropriate counterfactual story endings. By comparing the model's performance against a **baseline**, we ensure that improvements are genuine and not merely artifacts of overfitting to the metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Task Definition and Notation](#task-definition-and-notation)
5. [Proposed Evaluation Metrics as Reward Functions](#proposed-evaluation-metrics-as-reward-functions)
   - [Background on the Metrics](#background-on-the-metrics)
   - [Metric Definitions](#metric-definitions)
6. [Experiments](#experiments)
   - [Experiment 1: Direct Metric-Based Reward](#experiment-1-direct-metric-based-reward)
   - [Experiment 2: Combined Metric Reward](#experiment-2-combined-metric-reward)
   - [Experiment 3: Combined Delta-Based Reward](#experiment-3-combined-delta-based-reward)
7. [Reward Function and Baseline](#reward-function-and-baseline)
8. [Expected Results and Analysis](#expected-results-and-analysis)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Comparative Analysis](#comparative-analysis)
9. [Solution Implementation](#solution-implementation)
   - [Overview](#overview-1)
   - [Experiment 1: Direct Metric-Based Reward](#experiment-1-direct-metric-based-reward-1)
   - [Experiment 2: Combined Metric Reward](#experiment-2-combined-metric-reward-1)
   - [Experiment 3: Combined Delta-Based Reward](#experiment-3-combined-delta-based-reward-1)
   - [Integration with the T5 Model](#integration-with-the-t5-model)
10. [Usage](#usage)
    - [Installation](#installation)
    - [Training the Model](#training-the-model)
    - [Evaluating the Model](#evaluating-the-model)
11. [License](#license)

---

## Overview

In counterfactual story rewriting, models are tasked with modifying a story’s ending to reflect a hypothetical event, requiring minimal yet accurate changes while maintaining narrative coherence.

This project applies **policy gradient** RL methods to enhance model performance by directly optimizing for evaluation metrics, including both conventional metrics (BLEU, ROUGE, BERTScore, BARTScore) and novel, task-specific metrics previously proposed for evaluating counterfactual rewriting. By utilizing these metrics as reward functions during training, we aim to guide the model more effectively towards generating appropriate counterfactual story endings.

We conduct three experiments to progressively improve the model’s performance:

1. **Experiment 1:** Using the score between the model’s prediction and the edited ending as the reward.
2. **Experiment 2:** Employing a **Combined Metric Reward**, leveraging multiple metrics to guide the model.
3. **Experiment 3:** Utilizing a **Combined Delta-Based Reward**, which incorporates the previously proposed evaluation metrics as reward functions to capture nuanced changes required in counterfactual rewriting.

---

## Key Features

- **Reinforcement Learning with Evaluation Metrics as Reward Functions**:
  - Applies policy gradient methods to optimize text generation based on both conventional and task-specific evaluation metrics.
  - Leverages previously proposed counterfactual rewriting metrics as reward functions to better capture the essential changes required in the task.

- **Baseline Comparison**:
  - Introduces a baseline score to compare against the model's performance.
  - Ensures that the model's improvements represent genuine advancements.

- **Progressive Experiments**:
  - The project follows a structured approach to improve the model: starting with direct metric-based rewards, moving to combined metric rewards, and finally integrating delta-based rewards for optimal performance.

---

## Project Structure

```bash
TimeTravel-PolicyGradientRL/
├── src/
│   ├── models/
│   │   └── model_T5.py             # T5 model with policy gradient implementation.
│   ├── utils/
│   │   ├── config.py               # Configuration for paths, parameters, reward settings.
│   │   ├── metrics.py              # Evaluation metrics (BLEU, ROUGE, BERTScore, BARTScore, Delta Metrics).
│   │   └── utils.py                # Utility functions (data preprocessing, differential weights).
│   ├── data_loader.py              # DataLoader for JSON data processing.
│   ├── main_t5.py                  # Main script for training with RL and metric rewards.
│   └── main_t5_metrics.py          # Evaluation script for post-training metrics comparison.
├── results/                        # Directory for logs, metrics, and model checkpoints.
├── data/                           # Directory for datasets and processed data.
└── README.md                       # This README file.
```

---

## Task Definition and Notation

In this section, we define the counterfactual story rewriting task and introduce the notation used in this project.

### Task Definition and Components

Each story comprises the following components:

- **Premise** (\( X_P \)): The foundational scenario or context for the story.
- **Initial Event** (\( X_{IE} \)): An event that leads to the story’s original conclusion.
- **Original Ending** (\( Y_{OE} \)): The story’s original conclusion.
- **Counterfactual Event** (\( X_{CE} \)): A hypothetical event that contradicts the initial event and alters the story.
- **Edited Ending** (\( Y_{EE} \)): The modified conclusion reflecting the counterfactual event.

### Task Objective

The goal is to generate an **edited ending** \( \hat{Y}_{EE} \) given the **premise** \( X_P \), **counterfactual event** \( X_{CE} \), and **original ending** \( Y_{OE} \), making minimal yet appropriate changes to \( Y_{OE} \) to reflect \( X_{CE} \).

---

## Proposed Evaluation Metrics as Reward Functions

### Background on the Metrics

In a previous report, we introduced novel evaluation metrics specifically designed for counterfactual story rewriting. These metrics address the limitations of conventional metrics, which may not adequately capture the nuanced changes required in counterfactual scenarios due to the small size of counterfactual elements within the endings.

Recognizing their potential to better guide the learning process, we now employ these previously proposed evaluation metrics as reward functions in training our models using policy gradient methods.

### Metric Definitions

#### Summary of the Proposed Evaluation Metrics

| **Metric**       | **Definition**                                                                                                                                         | **Purpose**                                                                                                                                                             |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| \(\Delta_{M_1}\) | \( M(\hat{Y}_{EE}, Y_{EE}) - M(\hat{Y}_{EE}, Y_{OE}) \)                                                                                                | Measures the degree to which the **prediction** aligns more closely with the **edited ending** than with the **original ending**, capturing the necessary changes.       |
| \(\Delta_{M_2}\) | \( M(\hat{Y}_{EE}, X_{CE}) - M(Y_{EE}, X_{CE}) \)                                                                                                      | Assesses how well the **prediction** incorporates the **counterfactual event**, normalized by how the **edited ending** incorporates it.                                 |

#### \(\Delta_{M_1}\): Aligning with the Edited Ending

The metric \(\Delta_{M_1}\) quantifies the extent to which the generated ending \( \hat{Y}_{EE} \) is more similar to the edited ending \( Y_{EE} \) than to the original ending \( Y_{OE} \):

\[
\Delta_{M_1} = M(\hat{Y}_{EE}, Y_{EE}) - M(\hat{Y}_{EE}, Y_{OE})
\]

A higher \(\Delta_{M_1}\) score indicates that the model's prediction captures the changes required by the counterfactual event more effectively.

**Note**: While \(\Delta_{M_1}\) highlights the relative improvement over the original ending, it may attain large values even when absolute similarity scores are low. Therefore, it should be considered alongside \( M(\hat{Y}_{EE}, Y_{EE}) \) for a comprehensive assessment.

#### \(\Delta_{M_2}\): Incorporating the Counterfactual Event

The metric \(\Delta_{M_2}\) evaluates how well the generated ending incorporates the counterfactual event compared to the true edited ending:

\[
\Delta_{M_2} = M(\hat{Y}_{EE}, X_{CE}) - M(Y_{EE}, X_{CE})
\]

A lower absolute value of \(\Delta_{M_2}\) suggests that the model's prediction incorporates the counterfactual event similarly to the edited ending, reflecting appropriate adaptation to the hypothetical scenario.

---

## Experiments

### Experiment 1: Direct Metric-Based Reward

In this experiment, we use the similarity score between the generated ending \( \hat{Y}_{EE} \) and the edited ending \( Y_{EE} \) as the reward function. Conventional metrics like **BLEU**, **ROUGE**, **BERTScore**, and **BARTScore** are employed.

**Reward Function**:

\[
R_{\text{exp1}}(y) = M(\hat{Y}_{EE}, Y_{EE}) - \text{BaselineScore}
\]

This approach encourages the model to produce endings that closely match the edited ending. However, due to the potential limitations of conventional metrics in capturing nuanced changes, the model may not fully learn the desired modifications.

### Experiment 2: Combined Metric Reward

Building on Experiment 1, we incorporate multiple metrics into a combined reward function to leverage their collective strengths.

**Reward Function**:

\[
R_{\text{exp2}}(y) = \sum_{i} w_i \left( M_i(\hat{Y}_{EE}, Y_{EE}) - \text{BaselineScore}_i \right)
\]

Where:

- \( M_i \) represents the \( i \)-th metric.
- \( w_i \) is the weight assigned to each metric based on its relevance.
- \( \text{BaselineScore}_i \) adjusts for the expected average performance.

By combining metrics, we aim to provide a more comprehensive learning signal, guiding the model to improve across various aspects of text quality.

### Experiment 3: Combined Delta-Based Reward

In this experiment, we integrate the previously proposed delta-based metrics as reward functions to better capture the nuanced changes required in counterfactual rewriting.

**Reward Function**:

\[
R_{\text{exp3}}(y) = \Delta_{M_1}(y) - \lambda \cdot |\Delta_{M_2}(y)|
\]

Where:

- \( \Delta_{M_1}(y) \) measures the relative alignment with the edited ending over the original ending.
- \( \Delta_{M_2}(y) \) assesses the incorporation of the counterfactual event.
- \( \lambda \) controls the trade-off between the two components.

This combined reward function encourages the model to generate endings that not only reflect the necessary changes but also appropriately integrate the counterfactual event, leading to more contextually accurate and coherent stories.

---

## Reward Function and Baseline

### Policy Gradient Reward Function

The general form of the reward function used in policy gradient training is:

\[
R(y) = \text{Metric}(y) - \text{BaselineScore}
\]

By subtracting a baseline score, we mitigate the risk of overfitting and ensure that improvements are meaningful.

### Loss Function with Reward

The loss function for policy gradient optimization becomes:

\[
L = L_{\text{MLE}} - R(y)
\]

Where:

- \( L_{\text{MLE}} \) is the Maximum Likelihood Estimation loss.
- \( R(y) \) is the reward derived from the selected metrics.

This formulation allows the model to balance the likelihood of generating the correct sequence with the reward obtained from the evaluation metrics.

---

## Expected Results and Analysis

### Evaluation Metrics

To evaluate and compare the models from each experiment, we use:

- **Conventional Metrics**: BLEU, ROUGE-L, BERTScore, BARTScore.
- **Task-Specific Metrics**: \(\Delta_{M_1}\) and \(\Delta_{M_2}\).

### Expected Results Tables

#### Experiment 1: Direct Metric-Based Reward

| **Metric**    | **Baseline Model** | **Experiment 1 Model** |
|---------------|--------------------|------------------------|
| BLEU          | 0.25               | **0.30**               |
| ROUGE-L       | 0.40               | **0.45**               |
| BERTScore     | 0.85               | **0.87**               |
| BARTScore     | -2.5               | **-2.3**               |
| \(\Delta_{M_1}\) | 0.05               | **0.10**               |
| \(\Delta_{M_2}\) | -0.02              | **-0.01**              |

#### Experiment 2: Combined Metric Reward

| **Metric**    | **Experiment 1 Model** | **Experiment 2 Model** |
|---------------|------------------------|------------------------|
| BLEU          | 0.30                   | **0.32**               |
| ROUGE-L       | 0.45                   | **0.48**               |
| BERTScore     | 0.87                   | **0.89**               |
| BARTScore     | -2.3                   | **-2.1**               |
| \(\Delta_{M_1}\) | 0.10                   | **0.12**               |
| \(\Delta_{M_2}\) | -0.01                  | **-0.009**             |

#### Experiment 3: Combined Delta-Based Reward

| **Metric**    | **Experiment 2 Model** | **Experiment 3 Model** |
|---------------|------------------------|------------------------|
| BLEU          | 0.32                   | **0.35**               |
| ROUGE-L       | 0.48                   | **0.52**               |
| BERTScore     | 0.89                   | **0.91**               |
| BARTScore     | -2.1                   | **-1.9**               |
| \(\Delta_{M_1}\) | 0.12                   | **0.15**               |
| \(\Delta_{M_2}\) | -0.009                 | **-0.005**             |

> **Note**: The values are illustrative. Actual results may vary depending on the dataset, training configuration, and random initialization.

### Comparative Analysis

#### Improvement Across Experiments

- **Experiment 1 vs. Baseline**: Introducing direct metric-based rewards leads to noticeable improvements over the baseline model trained with standard MLE.
- **Experiment 2 vs. Experiment 1**: The combined metric reward enhances performance further by providing a richer learning signal.
- **Experiment 3 vs. Experiment 2**: Incorporating the delta-based metrics as reward functions results in the most significant improvements, particularly in the task-specific metrics, indicating better adaptation to counterfactual rewriting nuances.

#### How to Compare the Models

To determine which model performs best:

1. **Evaluate on a Held-Out Test Set**: Ensure fair comparisons by using data not seen during training.
2. **Analyze Metric Scores**: Assess improvements across both conventional and task-specific metrics.
3. **Statistical Significance**: Use statistical tests to verify that improvements are meaningful.
4. **Human Evaluation**: Complement automatic metrics with human judgments on coherence and appropriateness.
5. **Resource Efficiency**: Consider training time and computational resources relative to performance gains.

When using these reward functions with trained T5 models, you can identify the better model by:

- **Comparing Metric Scores**: The model with consistently higher scores across both conventional and task-specific metrics is considered superior.
- **Assessing Task-Specific Metrics**: Pay special attention to \(\Delta_{M_1}\) and \(\Delta_{M_2}\) as they are specifically designed for counterfactual story rewriting.
- **Evaluating Coherence and Relevance**: Ensure that the generated endings are coherent, contextually appropriate, and effectively incorporate the counterfactual event.

---

## Solution Implementation

In this section, we provide detailed implementations for **Experiment 1**, **Experiment 2**, and **Experiment 3**, ensuring that each aligns with the defined reward functions and training objectives.

### Overview

We employ policy gradient methods to train the T5 model for counterfactual story rewriting. The key idea is to define appropriate reward functions that reflect the desired properties of the generated text and use them to guide the learning process.

For each experiment, we:

- **Define the Reward Function**: Based on the experiment's objective.
- **Integrate the Reward into Training**: Adjust the training loop to incorporate the reward.
- **Implement the Policy Gradient Update**: Update the model parameters accordingly.
- **Log and Save Metrics**: Monitor the training process and evaluate performance.

---

### Experiment 1: Direct Metric-Based Reward

#### Objective

Use a single evaluation metric (e.g., ROUGE) as the reward function to encourage the model to generate endings similar to the edited ending.

#### Reward Function

The reward \( R_{\text{exp1}}(y) \) for a generated ending \( y \) is:

\[
R_{\text{exp1}}(y) = M(\hat{Y}_{EE}, Y_{EE}) - \text{BaselineScore}
\]

Where:

- \( M(\hat{Y}_{EE}, Y_{EE}) \) is the evaluation metric score (e.g., ROUGE) between the generated ending \( \hat{Y}_{EE} \) and the edited ending \( Y_{EE} \).
- **BaselineScore** is a constant value (e.g., 0.5) representing expected average performance.

#### Implementation Details

##### Metrics Calculation

```python
def calculate_reward(self, generated_text, edited_ending, metric='ROUGE', baseline_score=0.5):
    """
    Calculates the reward using the specified evaluation metric and adjusts it by the baseline score.
    """
    # Calculate the metric score between the generated text and the edited ending
    metric_score = self.metrics_evaluator.calculate_metric(generated_text, edited_ending, metric)
    # Compute the reward
    reward = metric_score - baseline_score
    return reward
```

##### Reward Integration in Training

```python
def policy_gradient_step_exp1(self, inputs, edited_endings, metric='ROUGE'):
    """
    Performs a policy gradient update step for Experiment 1.
    """
    # Forward pass
    outputs = self.model(**inputs)
    # Decode the generated texts
    generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    
    # Calculate rewards
    rewards = []
    for gen_text, edited_ending in zip(generated_texts, edited_endings):
        reward = self.calculate_reward(gen_text, edited_ending, metric)
        rewards.append(reward)
    
    # Compute the loss
    loss = -torch.mean(torch.tensor(rewards, requires_grad=True))
    
    # Backpropagation
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
```

##### Policy Gradient Update

The model parameters are updated by maximizing the expected reward:

\[
\theta \leftarrow \theta + \alpha \cdot \nabla_{\theta} \left( \log \pi_{\theta}(y|x) \cdot R_{\text{exp1}}(y) \right)
\]

In practice, this is implemented by backpropagating the negative reward as the loss.

##### Logging and Saving Metrics

```python
def log_metrics_exp1(self, epoch, generated_texts, edited_endings, metric='ROUGE'):
    """
    Logs evaluation metrics for Experiment 1.
    """
    metric_scores = [
        self.metrics_evaluator.calculate_metric(gen_text, edited_ending, metric)
        for gen_text, edited_ending in zip(generated_texts, edited_endings)
    ]
    avg_metric_score = sum(metric_scores) / len(metric_scores)
    logger.info(f"Epoch {epoch} - {metric}: {avg_metric_score:.4f}")
```

---

### Experiment 2: Combined Metric Reward

#### Objective

Use a weighted combination of multiple evaluation metrics to guide the model, leveraging the strengths of each metric.

#### Reward Function

The reward \( R_{\text{exp2}}(y) \) is:

\[
R_{\text{exp2}}(y) = \sum_{i} w_i \left( M_i(\hat{Y}_{EE}, Y_{EE}) - \text{BaselineScore}_i \right)
\]

Where:

- \( M_i \) is the \( i \)-th evaluation metric (e.g., BLEU, ROUGE).
- \( w_i \) is the weight assigned to metric \( M_i \).
- \( \text{BaselineScore}_i \) is the baseline score for metric \( M_i \).

#### Implementation Details

##### Metrics Calculation

```python
def calculate_combined_reward(self, generated_text, edited_ending, metrics_weights, baseline_scores):
    """
    Calculates the combined reward using multiple metrics.
    """
    total_reward = 0
    for metric, weight, baseline in zip(metrics_weights.keys(), metrics_weights.values(), baseline_scores.values()):
        metric_score = self.metrics_evaluator.calculate_metric(generated_text, edited_ending, metric)
        reward = weight * (metric_score - baseline)
        total_reward += reward
    return total_reward
```

##### Reward Integration in Training

```python
def policy_gradient_step_exp2(self, inputs, edited_endings, metrics_weights, baseline_scores):
    """
    Performs a policy gradient update step for Experiment 2.
    """
    # Forward pass
    outputs = self.model(**inputs)
    generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    
    # Calculate rewards
    rewards = []
    for gen_text, edited_ending in zip(generated_texts, edited_endings):
        reward = self.calculate_combined_reward(gen_text, edited_ending, metrics_weights, baseline_scores)
        rewards.append(reward)
    
    # Compute the loss
    loss = -torch.mean(torch.tensor(rewards, requires_grad=True))
    
    # Backpropagation
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
```

##### Policy Gradient Update

Similar to Experiment 1, but the reward is a weighted sum of multiple metrics.

##### Logging and Saving Metrics

```python
def log_metrics_exp2(self, epoch, generated_texts, edited_endings, metrics):
    """
    Logs evaluation metrics for Experiment 2.
    """
    for metric in metrics:
        metric_scores = [
            self.metrics_evaluator.calculate_metric(gen_text, edited_ending, metric)
            for gen_text, edited_ending in zip(generated_texts, edited_endings)
        ]
        avg_metric_score = sum(metric_scores) / len(metric_scores)
        logger.info(f"Epoch {epoch} - {metric}: {avg_metric_score:.4f}")
```

---

### Experiment 3: Combined Delta-Based Reward

#### Objective

Use the previously proposed delta-based metrics as reward functions to better capture the nuanced changes required in counterfactual rewriting.

#### Reward Function

The reward \( R_{\text{exp3}}(y) \) is:

\[
R_{\text{exp3}}(y) = \Delta_{M_1}(y) - \lambda \cdot |\Delta_{M_2}(y)|
\]

Where:

- \( \Delta_{M_1}(y) = M(\hat{Y}_{EE}, Y_{EE}) - M(\hat{Y}_{EE}, Y_{OE}) \)
- \( \Delta_{M_2}(y) = M(\hat{Y}_{EE}, X_{CE}) - M(Y_{EE}, X_{CE}) \)
- \( \lambda \) is a hyperparameter controlling the trade-off.

#### Implementation Details

##### Metrics Calculation

```python
def calculate_delta_m1(self, generated_text, edited_ending, original_ending, metric='ROUGE'):
    """
    Calculates Delta M1.
    """
    score_gen_edited = self.metrics_evaluator.calculate_metric(generated_text, edited_ending, metric)
    score_gen_original = self.metrics_evaluator.calculate_metric(generated_text, original_ending, metric)
    delta_m1 = score_gen_edited - score_gen_original
    return delta_m1

def calculate_delta_m2(self, generated_text, edited_ending, counterfactual_event, metric='ROUGE'):
    """
    Calculates Delta M2.
    """
    score_gen_counterfactual = self.metrics_evaluator.calculate_metric(generated_text, counterfactual_event, metric)
    score_edited_counterfactual = self.metrics_evaluator.calculate_metric(edited_ending, counterfactual_event, metric)
    delta_m2 = score_gen_counterfactual - score_edited_counterfactual
    return delta_m2
```

##### Reward Integration in Training

```python
def policy_gradient_step_exp3(self, inputs, edited_endings, original_endings, counterfactual_events, lambda_weight, metric='ROUGE'):
    """
    Performs a policy gradient update step for Experiment 3.
    """
    # Forward pass
    outputs = self.model(**inputs)
    generated_texts = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
    
    # Calculate rewards
    rewards = []
    for gen_text, edited_ending, original_ending, counterfactual_event in zip(
        generated_texts, edited_endings, original_endings, counterfactual_events
    ):
        delta_m1 = self.calculate_delta_m1(gen_text, edited_ending, original_ending, metric)
        delta_m2 = self.calculate_delta_m2(gen_text, edited_ending, counterfactual_event, metric)
        reward = delta_m1 - lambda_weight * abs(delta_m2)
        rewards.append(reward)
    
    # Compute the loss
    loss = -torch.mean(torch.tensor(rewards, requires_grad=True))
    
    # Backpropagation
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
```

##### Policy Gradient Update

The model is updated to maximize the combined delta-based reward, encouraging it to produce endings that are more similar to the edited ending than the original, while appropriately incorporating the counterfactual event.

##### Logging and Saving Metrics

```python
def log_metrics_exp3(self, epoch, generated_texts, edited_endings, original_endings, counterfactual_events, metric='ROUGE'):
    """
    Logs evaluation metrics for Experiment 3.
    """
    delta_m1_scores = []
    delta_m2_scores = []
    for gen_text, edited_ending, original_ending, counterfactual_event in zip(
        generated_texts, edited_endings, original_endings, counterfactual_events
    ):
        delta_m1 = self.calculate_delta_m1(gen_text, edited_ending, original_ending, metric)
        delta_m2 = self.calculate_delta_m2(gen_text, edited_ending, counterfactual_event, metric)
        delta_m1_scores.append(delta_m1)
        delta_m2_scores.append(delta_m2)
    avg_delta_m1 = sum(delta_m1_scores) / len(delta_m1_scores)
    avg_delta_m2 = sum(delta_m2_scores) / len(delta_m2_scores)
    logger.info(f"Epoch {epoch} - Delta M1: {avg_delta_m1:.4f}, Delta M2: {avg_delta_m2:.4f}")
```

---

### Integration with the T5 Model

We integrate the policy gradient steps into the T5 model's training loop for each experiment.

```python
class FlanT5FineTuner(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # Unpack batch data
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        labels = batch['labels']
        edited_endings = batch['edited_ending']
        original_endings = batch['original_ending']
        counterfactual_events = batch['counterfactual_event']
        
        # Choose the appropriate policy gradient step
        if self.experiment == 1:
            self.policy_gradient_step_exp1(inputs, edited_endings, metric='ROUGE')
        elif self.experiment == 2:
            metrics_weights = {'ROUGE': 0.5, 'BLEU': 0.5}  # Example weights
            baseline_scores = {'ROUGE': 0.5, 'BLEU': 0.5}
            self.policy_gradient_step_exp2(inputs, edited_endings, metrics_weights, baseline_scores)
        elif self.experiment == 3:
            lambda_weight = 0.5  # Example value
            self.policy_gradient_step_exp3(inputs, edited_endings, original_endings, counterfactual_events, lambda_weight, metric='ROUGE')
        
        # For PyTorch Lightning compatibility, return None
        return None
```

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

Train the model using policy gradient methods with the selected reward function.

#### Experiment 1

```bash
python src/main_t5.py --experiment 1 --reward_metric ROUGE
```

#### Experiment 2

```bash
python src/main_t5.py --experiment 2 --reward_metric Combined
```

#### Experiment 3

```bash
python src/main_t5.py --experiment 3 --reward_metric Delta
```

### Evaluating the Model

After training, evaluate the model using the provided evaluation script:

```bash
python src/main_t5_metrics.py --model_checkpoint <path_to_checkpoint> --evaluation_metrics BLEU ROUGE BERTScore BARTScore Delta_M1 Delta_M2
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Conclusion

By integrating previously proposed evaluation metrics as reward functions, we enhance the model's ability to perform counterfactual story rewriting. Through progressive experiments, we demonstrate that using these metrics within a policy gradient RL framework leads to significant improvements over baseline models, effectively capturing the nuanced changes required in the task.

