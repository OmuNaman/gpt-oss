"""
Downloads and tokenizes the TinyStories dataset.

This script follows the nanoGPT repository's data preparation methodology.
1.  Downloads the TinyStories dataset from Hugging Face.
2.  Tokenizes the text using the 'o200k_base' tokenizer via tiktoken.
3.  Concatenates all stories into a single stream of tokens, separated by <|endoftext|>.
4.  Splits the token stream into a training set (90%) and a validation set (10%).
5.  Saves the token arrays to binary files (`train.bin` and `val.bin`) in the
    same directory, ready to be used by the training script.
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

def main():
    # --- Configuration ---
    # The output directory will be the same as the script's directory
    output_dir = os.path.dirname(__file__)
    dataset_name = "roneneldan/TinyStories"
    split_ratio = 0.9

    # --- 1. Download the dataset ---
    print(f"Downloading dataset '{dataset_name}'...")
    # We use the 'train' split as it's the main one containing all stories.
    # We will create our own validation split from this.
    dataset = load_dataset(dataset_name, split="train")

    # --- 2. Initialize the tokenizer ---
    print("Initializing 'o200k' tokenizer...")
    enc = tiktoken.get_encoding("o200k_harmony")
    eot_token = enc.eot_token # The End-Of-Text token ID

    def tokenize_and_separate(example):
        """
        Tokenizes a single story and appends the EOT token.
        This ensures stories are treated as separate sequences by the model.
        """
        tokens = enc.encode_ordinary(example['text']) # Use encode_ordinary to avoid special tokens
        tokens.append(eot_token)
        return tokens

    # --- 3. Tokenize all stories and concatenate ---
    print("Tokenizing and concatenating all stories...")
    all_tokens = []
    # Using tqdm for a nice progress bar
    for story_tokens in tqdm(map(tokenize_and_separate, dataset), total=len(dataset)):
        all_tokens.extend(story_tokens)

    # Convert the Python list to a NumPy array for efficiency
    # Use uint16 since gpt2 vocab_size is 50257, which does not fits in 16 bits
    all_tokens_np = np.array(all_tokens, dtype=np.uint32)
    print(f"Total tokens in the dataset: {len(all_tokens_np):,}")

    # --- 4. Split into training and validation sets ---
    print(f"Splitting data into train ({split_ratio*100}%) and validation ({(1-split_ratio)*100}%) sets...")
    split_index = int(len(all_tokens_np) * split_ratio)
    train_tokens = all_tokens_np[:split_index]
    val_tokens = all_tokens_np[split_index:]

    print(f"Training set has {len(train_tokens):,} tokens.")
    print(f"Validation set has {len(val_tokens):,} tokens.")

    # --- 5. Save the token arrays to binary files ---
    train_output_path = os.path.join(output_dir, 'train.bin')
    val_output_path = os.path.join(output_dir, 'val.bin')

    print(f"Saving training tokens to '{train_output_path}'...")
    train_tokens.tofile(train_output_path)

    print(f"Saving validation tokens to '{val_output_path}'...")
    val_tokens.tofile(val_output_path)

    print("\nData preparation complete!")
    print("You can now run the training script pointing to this directory.")

if __name__ == "__main__":
    main()