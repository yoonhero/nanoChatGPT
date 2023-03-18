import glob
import pandas as pd
import json
import torch
from torch.utils.data import Dataset

from config import device
from tokenizer import tokenizer

# Data Loading Optimization
class GPTDataset(Dataset):
    def __init__(self, txt_file, block_size):
        self.block_size = block_size
        
        print(f"Loading Enormous Corpus Start...")
        with open(txt_file, "r") as f:
            self.tokens = f.read()[:100000000]
        print(f"Loading Corpus File Done!")

        print("Tokenizing...")
        enc = tokenizer
        self.encoded_token = enc.encode(self.tokens)
        self.length = len(self.encoded_token) // self.block_size
        print(f"Dataset Size: {len(self.encoded_token)}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx + 1) * self.block_size
        tokens = self.encoded_token[start_idx:end_idx]
        x = torch.tensor(tokens[:-1]).long()
        y = torch.tensor(tokens[1:]).long()
        
        x, y = x.to(device), y.to(device)
        return x, y
    

def create_dataset():
    files = glob.glob("./dataset/NIKLNEWSPAPER_2022_v1.0/*.json")
    result = ''
    for raw_data_path in files:
        with open(raw_data_path) as f:
            js = json.loads(f.read())
        df = pd.DataFrame(js["document"])

        paragraphs = df["paragraph"]
        sentences = [sentence["form"] for article in paragraphs for sentence in article ]

        result += "\n".join(sentences)

    with open("data.txt", "w") as f:
        f.writelines(result)


if __name__ == '__main__':
    create_dataset()
        
            
        