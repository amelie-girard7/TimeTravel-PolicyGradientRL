# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/data_loader.py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.utils.utils import preprocess_data, collate_fn
from src.utils.config import CONFIG

class CustomJSONDataset(Dataset):
    """
    A custom PyTorch Dataset class designed for loading and preprocessing data stored in JSON format.
    Supports tokenization and preprocessing for model training and evaluation.
    """
    def __init__(self, file_path, tokenizer, dataset_type="TimeTravel"):
        """
        Initializes the dataset object.

        Args:
            file_path (str): Path to the JSON file containing the data.
            tokenizer (T5Tokenizer): The tokenizer to use for preprocessing.
        """
        print(f"Initializing CustomJSONDataset with dataset_type: {dataset_type}")
        # Attempt to load and preprocess the data from the provided JSON file
        try:
            data = pd.read_json(file_path, lines=True)
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing {file_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

        self.tokenizer = tokenizer
        self.data_type = dataset_type

        # Preprocess each row of the data using the provided tokenizer
        self.processed_data = data.apply(lambda row: preprocess_data(row, self.tokenizer, dataset_type=dataset_type), axis=1, result_type='expand')
        print(f"First few rows of processed data: {self.processed_data.head()}")

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index from the dataset.
        
        Args:
            idx (int): The index of the item to retrieve.
        
        Returns:
            dict: A single data item, preprocessed and ready for model input.
        """
        item = self.processed_data.iloc[idx]
        return item

def create_dataloaders(data_path, tokenizer, batch_size, num_workers, dataset_type="TimeTravel"):
    """
    Creates DataLoader instances for each dataset specified by the configuration.

    Args:
        data_path (str): Path to the base directory containing data files.
        tokenizer (T5Tokenizer): The tokenizer to use for preprocessing the data.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of worker threads to use for loading data.
        dataset_type (str): Specifies the dataset tpe (eg. "ART", "TimeTravel","AblatedTimeTravel").

    Returns:
        dict: A dictionary of DataLoader objects, keyed by dataset type ('train', 'dev', 'test').
    """

    print(f"Creating dataloaders for dataset_type: {dataset_type}")  # Debug dataset type

    file_names = [CONFIG["train_file"], CONFIG["dev_file"], CONFIG["test_file"]]

    dataloaders = {}
    for file_name in file_names:
        file_path = Path(data_path) / file_name
        print(f"Loading file: {file_path} for dataset_type: {dataset_type}")  # Debug file path
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        # Create an instance of the dataset for each data file
        dataset = CustomJSONDataset(file_path, tokenizer, dataset_type=dataset_type)

        # Determine whether to shuffle: shuffle only for training
        shuffle = file_name == CONFIG["train_file"]

        # Create a DataLoader for each dataset
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id),
            num_workers=num_workers,
            shuffle=shuffle  # Shuffle data only for training
        )

        # Use the file name (without extension) as the key
        key = file_name.split('.')[0]  # e.g., 'train_supervised_small_sample'
        dataloaders[key] = dataloader

    return dataloaders
