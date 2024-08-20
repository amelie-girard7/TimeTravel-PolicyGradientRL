# Training Objectives and Evaluation Metrics for Counterfactual Story Rewriting

This repository contains the dataset and code accompanying the paper:

**Training Objectives and Evaluation Metrics for Counterfactual Story Rewriting**  
*Authors: Amelie Girard, Inigo Jauregi, Massimo Piccardi*  

Link to paper: [paper link](#)

## Table of Contents
- [Task Overview](#task-overview)
- [Story Components](#story-components)
- [Training Objective: Differential Token Weighting (DTW)](#training-objective-differential-token-weighting-dtw)
- [Custom Loss Function](#custom-loss-function)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Task Overview
The task involves rewriting the endings of stories when a counterfactual event is introduced. The model is trained to recognize and emphasize the differences between the original and edited story endings, making the new endings fit logically with the hypothetical scenario.

## Story Components
Each story consists of the following structured components:

- **Premise (𝑋𝑃):** Sets the foundational scenario or context for the story.
- **Initial Event (𝑋𝐼𝐸):** Introduces an event that leads to the original story's conclusion.
- **Original Ending (𝑋𝑂𝐸):** The original conclusion of the story.
- **Counterfactual Event (𝑋𝐶𝐸):** A divergent hypothetical event that alters the course of the story.

## Training Objective: Differential Token Weighting (DTW)
The conventional training approach treats all words equally. However, our task requires the model to focus more on the differences between the original and edited endings. We achieve this through Differential Token Weighting (DTW), where higher weights are assigned to tokens that differ between the original and edited endings.

### Tools Used
- **spaCy:** For semantic similarity checks to detect contextually similar or different terms.
- **NLTK:** For synonym checks using WordNet to capture subtle variations.

## Custom Loss Function
We implemented a custom loss function that incorporates the differential weights, ensuring that the model learns to emphasize critical differences between the original and edited endings.

## Dataset
The **TimeTravel dataset** is used for training and evaluation. This dataset contains structured stories with both original and edited endings:

- **Training Set:** 16,752 stories
- **Validation Set:** 1,871 stories
- **Test Set:** 1,871 stories
- **Gold Standard Set:** 604 manually curated stories for high-quality evaluation

## Evaluation Metrics
To assess the performance of the model, we use the following metrics:

- **BARTScore**
- **ROUGE-L**
- **BERTScore**
- **SacreBLEU**
- **Counterfactual Rewriting Metrics** (custom metric)

These metrics help evaluate the coherence, relevance, and adaptability of the generated counterfactual story endings.


## Repository Structure

The project is organized into the following structure:

```plaintext
TIMETRAVEL/
├── bertviz/                 # Customized BERTviz for visualizing attention in Transformer models.
├── data/                    # Directory for storing dataset files.
├── models/                  # Directory for storing trained models or model-related files.
└── src/                     # Main source code directory.
    ├── BARTScore_metric/    # Contains the implementation of the BARTScore metric.
    │   ├── models/          # Model-related scripts, including model_T5.py for T5 model implementation.
    │   └── utils/           # Utility scripts including configuration, metrics, and helper functions.
    ├── __init__.py          # Initialization file for the src module.
    ├── data_loader.py       # Script for loading and preprocessing data.
    ├── .gitignore           # Git ignore file specifying files and directories to ignore.
    ├── generate_attentions.py  # Script to generate attention visualizations.
    ├── LICENSE              # License file for the project.
    ├── main_gpt.py          # Main script to run the GPT model.
    ├── main_t5.py           # Main script to run the T5 model.
    ├── README.md            # This README file.
    ├── requirements.txt     # List of dependencies required for the project.
    └── vis_attention_T5.ipynb  # Jupyter notebook for visualizing attention in the T5 model.
```

## Usage

### Training the Model
To train the model, ensure that the TimeTravel dataset is properly set up. The model can be trained using the provided scripts, with options to adjust the hyperparameters, including the differential weights.

### Evaluating the Model
The evaluation scripts allow you to assess the model's performance using the metrics mentioned above. Multiple edited endings are evaluated against the counterfactual scenarios to ensure robust performance.

## Installation
To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/counterfactual-story-rewriting.git
cd counterfactual-story-rewriting
pip install -r requirements.txt

```

## Contributing

We welcome contributions from the community to improve this project. If you'd like to contribute, please follow these steps:

1. **Fork the repository.**  
   This creates a copy of the project under your GitHub account.

2. **Create a new branch for your feature or bugfix.**  
   Use a descriptive name for your branch (e.g., `feature/add-new-metric` or `bugfix/fix-token-weighting`).

3. **Make your changes.**  
   Implement your feature or bugfix, and ensure your code is well-documented and tested.

4. **Submit a pull request.**  
   Go to the repository's GitHub page and open a pull request. Provide a clear and detailed description of your changes, why they were necessary, and any relevant context.

### Reporting Issues

If you encounter any issues or bugs, please open an issue in the repository. When reporting issues, include as much detail as possible to help us address the problem quickly:

- Describe the problem and how to reproduce it.
- Provide any relevant error messages or screenshots.
- Mention the environment you're using (e.g., operating system, Python version).

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software under the terms of the license. See the [LICENSE](LICENSE) file for details.
