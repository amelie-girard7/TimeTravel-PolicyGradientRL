#/src/data_loader.py 

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from src.utils.utils import preprocess_data, collate_fn
from src.utils.config import CONFIG

class CustomJSONDataset(Dataset):
    """
    A custom PyTorch Dataset class designed for loading and preprocessing data stored in JSON format.
    It supports tokenization and other preprocessing steps necessary for model training and evaluation.
    """
    def __init__(self, file_path, tokenizer):
        """
        Initializes the dataset object.
        """
        # Attempt to load and preprocess data,
        try:
            data = pd.read_json(file_path, lines=True)
            # Reading data using pandas.
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing {file_path}: {e}")
            # Specific catch for parsing errors to provide more informative feedback.
        except FileNotFoundError:
            # Specific catch for file not found errors.
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.tokenizer = tokenizer # Store the tokenizer for later use in preprocessing.

        # Preprocess the data using the provided tokenizer and calculate differential weights
        # Applying preprocessing row-wise and expanding the result to columns for easier access
        self.processed_data = data.apply(lambda row: preprocess_data(row,self.tokenizer), axis=1, result_type='expand')
        print(self.processed_data.head())  # Debug print to check the structure of the processed data

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index from the dataset.
        
        Parameters:
            idx (int): The index of the item to retrieve.
            
        Returns:
            A single data item processed and ready for model input.
        """
        item = self.processed_data.iloc[idx]
        return item

def create_dataloaders(data_path, tokenizer, batch_size, num_workers):
    """
    Creates DataLoader instances for each dataset specified by file_names.
    
    Parameters:
        data_path (Path or str): The base path where data files are stored.
        tokenizer: The tokenizer to use for preprocessing text.
        batch_size (int): The number of items per batch.
        num_workers (int): The number of subprocesses to use for data loading.
        
    Returns:
        A dictionary of DataLoader objects keyed by file name.
    """
    file_names = [CONFIG["train_file"], CONFIG["dev_file"], CONFIG["test_file"]]
    
    dataloaders = {}
    for file_name in file_names:
        file_path = Path(data_path) / file_name
        print(f"Checking file: {file_path}")  # Debug print to check paths
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist.")
        dataset = CustomJSONDataset(file_path, tokenizer)

        # Creating an instance of the custom dataset for each file.
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, pad_token_id=tokenizer.pad_token_id), 
            num_workers=num_workers,
        )

        # Use a simplified key for each dataloader, based on the file name.
        key = file_name.split('.')[0]  # 'train_supervised_small_sample' for example
        dataloaders[key] = dataloader
        
    return dataloaders
