# Enhancing Counterfactual Story Rewriting via Policy Gradient Optimization

## Overview
This project explores Reinforcement Learning (RL), specifically policy gradient (PG) methods, to improve model performance in counterfactual story rewriting tasks. We compare Maximum Likelihood Estimation (MLE) training alone versus a combination of MLE and PG optimization. This approach aims to enhance coherence and contextual relevance in story endings that incorporate specified counterfactual events.

## Experimental Structure

### Experiments

- **Experiment 1**: Pure MLE Training
  - The model is trained exclusively using MLE for 6 epochs, and the best model checkpoint is saved based on validation loss.
  
- **Experiment 2**: Mixed MLE + PG Training
  - **Phase 1 (MLE Training)**: Train the model with MLE for 3 epochs, saving checkpoints throughout training.
  - **Phase 2 (PG Fine-Tuning)**: Fine-tune the MLE-trained model with PG for an additional 3 epochs. During this phase, reward-based optimization guides the model towards improved coherence and counterfactual alignment in the generated story endings.

### Task Definition and Annotation Schema

#### Task

The counterfactual story rewriting task requires generating an alternative story ending based on a hypothetical event (counterfactual) that contradicts the story’s initial event. The model is tasked with:
1. Integrating the counterfactual event into the ending,
2. Maintaining coherence and relevance in the new context, and
3. Making minimal changes to the original ending while incorporating the counterfactual.

The input for each story includes four components:
1. **Premise** (\( X_P \)): The foundational story context.
2. **Initial Event** (\( X_{IE} \)): The event leading to the original ending.
3. **Original Ending** (\( Y_{OE} \)): The story’s original conclusion.
4. **Counterfactual Event** (\( X_{CE} \)): A hypothetical event that contradicts the initial event.

The model generates an **Edited Ending** (\( \hat{Y}_{EE} \)) that adheres to the counterfactual event, ensuring story coherence and minimal alteration of the original ending.

#### Annotation Schema

The dataset includes pairs of **Original Endings** and **Edited Endings** based on specified counterfactual events. These annotations are formatted as follows:

- **Input Annotation**:
  - A concatenated sequence of the **Premise, Initial Event, Original Ending, and Counterfactual Event**.
  
- **Output Annotation**:
  - The target sequence is the **Edited Ending**, reflecting the modified story ending based on the counterfactual.

These annotated pairs serve two purposes:
1. **MLE Training Phase**: They provide supervised training data.
2. **Policy Gradient Fine-Tuning**: The reference Edited Ending is used to calculate reward scores, guiding the model towards alignment with the desired outcome during PG fine-tuning.

## Model and Training Phases

The T5 model is used for sequence generation and trained in distinct phases for each experiment:

1. **MLE Training Phase (Experiment 1)**: The model is trained in a supervised manner using cross-entropy loss. This phase aims to align generated endings with reference annotations, setting the baseline performance.
  
2. **Policy Gradient Fine-Tuning Phase (Experiment 2)**: The model is further fine-tuned using PG, optimizing for maximum reward based on selected evaluation metrics. This phase enhances alignment with the counterfactual scenario and improves the quality of generated story endings.

## Reward Metrics in Policy Gradient Training

During PG fine-tuning, rewards are calculated based on multiple evaluation metrics. These metrics capture various aspects of textual coherence, fluency, and alignment with the reference endings. The reward score for each generated ending is derived by comparing it to the annotated edited ending, encouraging the model to produce outputs that closely match the target. The reward is adjusted by a baseline score to stabilize training.

### Metrics Used in Reward Calculation

1. **ROUGE-L**
   - **Description**: Measures the longest common subsequence (LCS) between the generated and reference text, assessing fluency and structural similarity.
   - **Reward Calculation**: Provides a reward score based on LCS overlap, rewarding endings that structurally align with the reference.
   - **Role in PG**: Encourages the model to maintain narrative structure and phrase similarity with the reference edited ending.

2. **BERTScore**
   - **Description**: Computes the semantic similarity of tokens between generated and reference texts using BERT embeddings, capturing deeper meaning beyond surface overlap.
   - **Reward Calculation**: Assigns rewards based on token-level embedding similarity, accounting for synonyms and semantically equivalent phrasing.
   - **Role in PG**: Helps the model prioritize semantic coherence, ensuring that the generated ending aligns with the story’s thematic content.

3. **BARTScore**
   - **Description**: Uses a BART model to evaluate the coherence and relevance of generated text in comparison to the reference, measuring fluency and logical flow.
   - **Reward Calculation**: Scores the generated text based on its probability under BART, rewarding coherent sequences.
   - **Role in PG**: Guides the model to produce grammatically sound, contextually appropriate sequences that contribute to a smooth story progression.

4. **SacreBLEU**
   - **Description**: Measures precision by comparing n-gram overlap between generated and reference texts, with adjustments for brevity.
   - **Reward Calculation**: Generates a reward based on n-gram precision, rewarding the generated text for capturing key phrases from the reference.
   - **Role in PG**: Incentivizes accuracy in capturing specific content and phrases from the reference edited ending.

5. **Counterfactual Reward Metrics (CRMs)**
   - **Description**: Task-specific metrics used to capture alignment with counterfactual elements.
   - **Metric Formulation**:
     - \( \Delta_{M_1} \): Measures the difference between the generated ending \( \hat{Y}_{EE} \) and both the edited ending \( Y_{EE} \) and the original ending \( Y_{OE} \).
     - \( \Delta_{M_2} \): Evaluates the degree to which the generated ending incorporates the counterfactual event \( X_{CE} \).
   - **Role in PG**: Directly rewards endings that accurately reflect the counterfactual context, guiding the model to adhere closely to the hypothetical scenario introduced by the counterfactual event.

## Policy Gradient Training: Detailed Steps

During PG fine-tuning, the model is trained to maximize expected rewards by adjusting the log probability of the generated sequence based on its reward score. This process includes the following steps:

1. **Input Preparation and Preprocessing**:
   - Each input consists of a structured sequence combining the premise, initial event, original ending, and counterfactual event. This sequence is tokenized to produce input IDs and attention masks.

2. **Forward Pass and Logits Generation**:
   - In the forward pass, the model generates logits for each token in the vocabulary. These logits are transformed into probabilities to guide token selection in the generated sequence.

3. **Log Probability Calculation**:
   - After converting logits to probabilities, log probabilities are computed for each token, facilitating stable gradient calculations.

4. **Reward Calculation**:
   - Each generated sequence is scored against the reference edited ending using the selected metrics, and these reward scores are adjusted by a baseline to stabilize training.

5. **Policy Gradient Loss Calculation**:
   - The model’s loss is calculated based on the sequence’s log probability and weighted by the reward. This encourages the model to learn and maximize outputs that receive high reward scores, aligning with the target reference.

## Experimental Setup

### Configuration

Configuration parameters are defined in a configuration file (`config.py`). These include:
- Number of epochs for each phase (MLE and PG),
- Reward metrics,
- Baseline adjustments for PG, and
- Training parameters like batch size and learning rate.

### Running the Experiment

1. **MLE Training (Experiment 1)**: Run `main_mle.py` to perform MLE training and save the best model checkpoint based on validation loss.
   
2. **PG Fine-Tuning (Experiment 2)**: After completing MLE training, run `main_pg.py` to load the best MLE checkpoint and apply PG fine-tuning.

