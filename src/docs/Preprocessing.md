The architecture comprises three main components:

1. **Custom Dataset Class**: Handles loading and preprocessing the data.
2. **Utility Functions**: Provides functions for counting lines, loading data, and preprocessing data.
3. **Data Loader Creation Function**: Creates data loaders for different datasets (training, validation, testing).

### 1. Custom Dataset Class (`CustomJSONDataset`)

This class is responsible for:
- Loading the JSON data.
- Preprocessing the data using a tokenizer.
- Providing methods to access the length of the dataset and individual items.

**Key Methods:**
- **`__init__`**: Initializes the dataset by loading and preprocessing the JSON data.
- **`__len__`**: Returns the number of items in the dataset.
- **`__getitem__`**: Retrieves an individual data item by index.

### 2. Utility Functions

These functions provide additional functionality required by the dataset and data loader creation process.

**Functions:**
- **`count_json_lines`**: Counts the number of lines in a JSON file, useful for estimating dataset size.
- **`load_first_line_from_json`**: Loads and parses the first line from a JSON file to inspect data structure.
- **`preprocess_data`**: Preprocesses a single data row by tokenizing text fields and calculating differential weights.
- **`calculate_differential_weights`**: Calculates weights for tokenized labels based on differences, which adjusts the model's learning focus.
- **`collate_fn`**: Collates a batch of preprocessed data, including padding sequences to equalize their lengths.

### 3. Data Loader Creation Function (`create_dataloaders`)

This function creates data loaders for the datasets specified in the configuration.

**Steps:**
1. **Define File Names**: Gets the file names for training, validation, and testing datasets from the configuration.
2. **Initialize Data Loaders Dictionary**: Initializes an empty dictionary to store data loaders.
3. **Load and Process Each Dataset**:
   - **Check File Existence**: Ensures the data file exists.
   - **Create Dataset Instance**: Creates an instance of `CustomJSONDataset`.
   - **Create Data Loader**: Creates a data loader for the dataset with specified batch size and number of workers.
   - **Store in Dictionary**: Stores the data loader in the dictionary with a key based on the file name.

### Interaction Flow

1. **Loading Data**:
   - The `create_dataloaders` function reads the data file paths and initializes data loaders for each dataset (train, validation, test).

2. **Custom Dataset Initialization**:
   - For each dataset file, an instance of `CustomJSONDataset` is created. This involves loading the JSON data using pandas and preprocessing it using the `preprocess_data` function.
   - During preprocessing, the `calculate_differential_weights` function is called to adjust token weights based on the differences in the text.

3. **Data Loader Creation**:
   - The `DataLoader` instance is created for each dataset, using the `collate_fn` to handle batch collation, including padding sequences to ensure they have the same length.

4. **Batch Processing**:
   - When the data loader fetches a batch of data, it uses the `__getitem__` method of `CustomJSONDataset` to retrieve individual items.
   - The `collate_fn` pads sequences and prepares the batch for model input.

### Mathematical Equations

**Padding Sequences**:
- Padding ensures all sequences in a batch have the same length. If we have sequences of lengths \( l_1, l_2, \ldots, l_n \), the padded length \( L \) is the maximum sequence length in the batch:
  \[
  L = \max(l_1, l_2, \ldots, l_n)
  \]

**Differential Weights Calculation**:
- For each token in the tokenized labels, if the token matches a token from the differences list, its weight is set to a high value (\( w_{\text{high}} \)), otherwise to a base value (\( w_{\text{base}} \)).

