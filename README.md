# Detecting Hate Speech Model for ViHOS Dataset

This repository contains code for training and evaluating a custom machine learning model for detecting hate speech on a sentence level, based on the `transformers` library and designed to work with the ViHOS-formatted dataset.


## Installation

To run this project, you can use Google Colab or Kaggle, which provide free access to GPUs and the necessary libraries pre-installed.


1. Clone the repository by running the following command in a code cell:
   ```python
   !git clone https://github.com/Ndphuong-17/ViHOS_chunking.git
   %cd ViHOS_chunking
   ```

2. Install the required libraries (if needed) using:
   ```python
   !pip install transformers torch
   ```

3. Adjust the paths to the dataset files on your environment, and run the script as explained below.

## Usage

To run the training and testing process, use the following command:

```bash
!python main.py --train_path "Data\Chunking_data\train.csv" --dev_path "Data\Chunking_data\dev.csv" --test_path "Data\Chunking_data\test.csv" --batch_size 64 --max_len 128 --lr 5e-6 --num_epochs 2 --output_json "test_results.json" --output_dir "output"
```

### Command Line Arguments

- `--train_path`: Path to the training data CSV file.
- `--dev_path`: Path to the validation data CSV file.
- `--test_path`: Path to the testing data CSV file.
- `--test_index`: Index of the test data to be evaluated.
- `--batch_size`: Number of samples per gradient update.
- `--max_len`: Maximum sequence length for the input data.
- `--lr`: Learning rate for the optimizer.
- `--num_epochs`: Number of training epochs.
- `--output_json`: File name for saving the test results in JSON format.
- `--output_dir`: Directory to save the output model and results.

