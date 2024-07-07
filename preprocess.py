# Libraries needed
import pandas as pd
import re
from nltk import RegexpTokenizer
from sklearn.model_selection import train_test_split
from typing import List
import os

# Initialization of tools
tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')


def preprocess_text(text: str):
    if not isinstance(text, str): # Pre-condition
        raise ValueError("Expected a string for text preprocessing")
    text = text.lower()  # All characters in lowercase
    # Remove numbers, extra whitespaces and punctuation
    text = re.sub(r'\b\d+\b', '', text)  # Numbers
    text = re.sub(r'\s+', ' ', text)  # Extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Punctuation
    return text


def tokenize_text(text: str):
    if not isinstance(text, str): # Pre-condition
        raise ValueError("Expected a string for text tokenization")
    # Tokenization
    return tokenizer.tokenize(text)

# Function for applying both preprocessing and tokenization to a Pandas df
def preprocess_and_tokenize(df: pd.DataFrame, text_column: str):
    df['processed_text'] = df[text_column].apply(preprocess_text)
    df['processed_text'] = df['processed_text'].apply(tokenize_text)
    return df

# Save the dataset, I use parquet because is the less memory-consuming format
def save_dataframe_as_parquet(df: pd.DataFrame, file_path: str):
    # Pre-condition
    output_dir = os.path.dirname(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_parquet(file_path, index=False)
    print(f"File saved to '{file_path}'")



# Define the path to the data
file_path = './data/train.jsonl'

# Read the data
df = pd.read_json(file_path, lines=True)

# Verify that the 'text' and 'label' columns exist
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("The DataFrame does not contain the necessary 'text' and 'label' columns.")

# Preprocess and Tokenize the text column
df = preprocess_and_tokenize(df, 'text')



X = df['processed_text'].values
y = df['label'].values

# Train-test split, random state used for repetibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2024)

# Combine X and y into a list of dictionaries for train and text
train_data = [{'processed_text': ' '.join(text), 'label': label} for text, label in zip(X_train, y_train)]
test_data = [{'processed_text': ' '.join(text), 'label': label} for text, label in zip(X_test, y_test)]

# Convert the list of dictionaries to a DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Save the DataFrame to a Parquet file
save_dataframe_as_parquet(train_df, './preprocessed_and_tokenized_data/preprocessed_train_dataset.parquet')
save_dataframe_as_parquet(test_df, './preprocessed_and_tokenized_data/preprocessed_test_dataset.parquet')

