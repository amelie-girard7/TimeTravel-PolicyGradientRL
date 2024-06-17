# Evaluation Metrics for Counterfactual Story Re-writing

This repository contains the dataset and code accompanying the paper:

**"Title of Your Paper"**  
*Authors: Amelie Girard, Author2, Author3*  

Link to paper: [paper link](#)


## Repository Structure

The repository is structured as follows:

```
TIMETRAVEL/
│
├── README.md            # Project overview and instructions
├── LICENSE              # License information
├── .gitignore           # Files and folders to be ignored by git
│
├── src/                 # Source code for the project
│   ├── main.py          # Main script to run the models
│   ├── models/          # Model definitions
│   │   └── model_T5.py  # T5 model for story rewriting
│   ├── utils/           # Utility scripts and functions
│   │   └── utils.py     # Helper functions
│   └── data_loader.py   # Data loading and preprocessing scripts
│
├── data/                # Data directory
│   ├── raw/             # Raw data, unprocessed
│   ├── processed/       # Processed data ready for model input
│   └── external/        # Any external data sources
│
├── notebooks/           # Jupyter notebooks for experiments and analysis
│   └── exploration.ipynb
│
├── requirements.txt     # Project dependencies
│
├── tests/               # Test scripts
│   └── test_model_T5.py # Test for T5 model
│
├── scripts/             # Utility scripts, e.g., setup or installation scripts
│   └── setup.sh
│
├── models/              # Trained model files
│   └── model_T5.pkl
│
└── docs/                # Documentation files
    └── model_T5.md      # Documentation specific to T5 model
```

## Dataset: TimeTravel

The TimeTravel dataset is curated to facilitate the training and evaluation of models on the task of counterfactual story rewriting. It contains stories along with their original and counterfactually altered endings [cite the orginal paper].

- `train_supervised_small.json`: Supervised training set with human-annotated rewritten endings (28,363 examples).
- `train_supervised_large.json`: A larger version of the supervised training set (16,752 examples).
- `dev_data.json`: Development set (1,871 examples).
- `test_data.json`: Test set (1,871 examples).

The dataset can be **downloaded** from [here](https://drive.google.com/file/d/150jP5FEHqJD3TmTO_8VGdgqBftTDKn4w/view?usp=sharing).

### Data Example

The dataset includes stories. Here's how this looks in the dataset:

```json
{
  "story_id": "4fd7d150-b080-4fb1-a592-8c27fa6e1fc8",
  "premise": "Andrea wanted a picture of her jumping.",
  "initial": "She set the camera up.",
  "counterfactual": "She asked her friend to draw one.",
  "original_ending": "Then, she jumped in the air. The picture kept coming out wrong. It took twenty tries to get it right.",
  "edited_ending": [
    "Then, she jumped in the air to demonstrate how she wanted it to look.",
    "The picture kept coming out wrong.",
    "It took drawing it several times to get it right."
  ]
}
```

## Solution Architecture

The project is a structured Python application primarily dealing with data processing, model training, and prediction using a pre-trained Flan-T5 model. This is a breakdown of the functionality of each file and data flow:

### `src/main.py`
The main executable script for the project:

- Orchestrates the process by setting up the model, data loaders, and the trainer.
- Initializes the `FlanT5FineTuner` and prepares the data loaders for training, validation, and testing datasets.
- Sets up a PyTorch Lightning `Trainer` to manage the training process, including checkpointing and logging.
- Handles the execution of the training process and, optionally, testing and validation.

![Detailed project structure](./images/structure-1.png)
*Diagram showing the detailed structure of the project, including various components and their interactions.*

### `src/models/model_T5.py`
Defines the model and training procedures:
The core of this project is evaluating the causal reasoning of pre-trained language models, such as Flan-T5, for the task of counterfactual story rewriting and comparing it to GPT. The model is trained to minimize the log-likelihood of generating the actual rewritten endings based on the given story context and a counterfactual premise.

- **FlanT5FineTuner**: A class that wraps around the T5 model for fine-tuning on the story rewriting task. It includes methods for the forward pass, training, validation, and testing steps. It also includes methods for generating text (`generate_text`) and calculating custom metrics (`_calculate_metrics`).

### `src/utils/utils.py`
This file contains utility functions to handle data preprocessing and loading.

![Project structure overview](./images/structure.png)
*Diagram showing the high-level structure of the project.*

- **`count_json_lines(file_path)`**: Counts lines in a JSON file used for data validation or insight.
- **`load_first_line_from_json(file_path)`**: Loads the first line from a JSON file, which could be used for testing or data inspection.
- **`preprocess_data(row, tokenizer)`**: Processes each row of data, extracting necessary fields and constructing input-output sequences for model training or inference.
    - **Input**: Combines the `premise`, `initial`, `original_ending`, and `counterfactual` into a single input sequence.
    - **Output**: Tokenizes the `edited_ending` for model training.
- **`calculate_differential_weights(tokenized_labels, tokenizer, differences, high_weight=20, base_weight=1)`**: Calculates differential weights for tokenized labels based on differences.
- **`collate_fn(batch, pad_token_id=0, attention_pad_value=0)`**: Prepares batches of data by tokenizing and structuring them in a format that the model expects.

### `src/data_loader.py`
Handles data loading:

- **CustomJSONDataset**: A PyTorch `Dataset` class that reads data from a JSON file and preprocesses it using `preprocess_data` from `utils.py`.
- **`create_dataloaders(data_path, tokenizer, batch_size, num_workers)`**: Creates PyTorch `DataLoader` objects for batching and efficient data loading during model training or inference.

### Data Reading & Preprocessing
- Data is read from JSON files using the `CustomJSONDataset` class.
- Data is preprocessed per row using `preprocess_data` which constructs input-output sequences needed by the model.

### Data Batching & Tokenization
- `DataLoader` objects are created for batching.
- Batches of data are tokenized and structured properly by `collate_fn` during the data loading process.

### Model Training & Validation
- `FlanT5FineTuner` handles the model training, validation, and testing.
- It uses the batches prepared by `DataLoader` and performs forward passes, loss calculation, and backpropagation.
- It also generates text and calculates custom metrics like BLEU and ROUGE scores.

### Checkpointing & Logging
- Model checkpoints and logs are managed by PyTorch Lightning's `Trainer`, saving the state of the model and logging metrics for monitoring.

### Execution Control
- `main.py` orchestrates the whole process, ensuring that the model, data, and training utilities are correctly set up and executed.


