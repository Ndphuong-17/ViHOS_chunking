import pandas as pd
from sklearn.model_selection import train_test_split
import torch 


def split_path(path, test_index, train_path, dev_path, test_path):
    # Load the data
    data = pd.read_csv(path) # Replace 'data.csv' with your file path

        # Get a list of unique sentence IDs
    unique_sentence_ids = data['sentence_id'].unique()[:test_index] 

    # Split sentence IDs into train (70%), temp (30%)
    train_ids, temp_ids = train_test_split(unique_sentence_ids, test_size=0.30, random_state=42)

    # Further split the temp IDs into test (15%) and dev (15%)
    test_ids, dev_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    # Create subsets for each split using the sentence IDs
    train_data = data[data['sentence_id'].isin(train_ids)]
    test_data = data[data['sentence_id'].isin(test_ids)]
    dev_data = data[data['sentence_id'].isin(dev_ids)]


    # Save the splits into separate CSV files
    train_data[:-1].to_csv(train_path[:-4] + '_test.csv', index=False)
    test_data[:-1].to_csv(test_path[:-4] + '_test.csv', index=False)
    dev_data[:-1].to_csv(dev_path[:-4] + '_test.csv', index=False)

    train_path = train_path[:-4] + '_test.csv'
    test_path = test_path[:-4] + '_test.csv'
    dev_path = dev_path[:-4] + '_test.csv'

    print(f"Training set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Development set: {len(dev_data)} samples")

    return train_path, dev_path, test_path



def prepare_data(file_path):
    df = pd.read_csv(file_path)

    # remove nan
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"Columns: {df.columns}")

    texts = df['Chunk'].tolist()
    binary_spans = df['Tag'].tolist()

    return texts, binary_spans

# Dataloader function
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, spans, tokenizer, max_len = 128):
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length = max_len, truncation=True,
                                return_tensors="pt")for text in texts]


        self.spans = torch.tensor(spans)

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        return self.texts[index], self.spans[index]

def create_dataloader(data_path, batch_size, tokenizer, max_len, shuffle=True):
    dataset = TextDataset(*prepare_data(data_path), tokenizer, max_len)
    # return texts
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader