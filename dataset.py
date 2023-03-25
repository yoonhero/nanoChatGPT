import glob
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import gzip
import numpy as np
from transformers import AutoTokenizer
# from Korpora import Korpora

from config import device

# class NamuWikiDataset(Dataset):
#     def __init__(self, tokenizer, block_size):
#         # loading the namuwiki dataset
        

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):

def encode_tokens_from_file(file_path, eos_token, tokenizer)

class TokenedDataset(Dataset):
    def __init__(
            self, 
            file_path:str, 
            tokenizer:AutoTokenizer, 
            block_size:int, 
            encoding:str, 
            from_cache:bool=False, 
            line_by_line:bool=False, 
            save_cache: bool=False, 
            cache_destination: str = "dataset_cache.tar.gz",
            device:str="cuda"
        ):
        self.line_by_line = line_by_line
        self.device = device

        if from_cache:
            open_func = gzip.open if file_path.endswith(".gz") else open

            with open_func(file_path, "rb") as f:
                self.tokens = np.load(f)
            self.num_subsets = self.tokens.shape[0] - block_size
            self.block_size = block_size
            self.line_by_line = line_by_line
            return

        self.tokens = encode_tokens_from_file(
                file_path,
                eos_token,
                tokenizer,
                text_delim,
                header,
                progress_bar_refresh_rate,
        )



        if save_cache:
            self.save(cache_destination)

    def save_cache(self, cache_destination):
        with gzip.open(cache_destination, "wb") as f:
            np.save(f, self.tokens)

    def __len__(self):
        return self.num_subsets

    def __getitem__(self, idx):
        return torch.as_tensor(
            self.tokens[idx: (idx+self.block_size)].astype(np.int64, copy=False),
            dtype=torch.long,
            device=self.device
        )

    def __repr__(self) -> str: 
        return f"TokenDataset containing {self.num_subsets} subsets."


# Data Loading Optimization
class GPTDataset(Dataset):
    def __init__(self, txt_file, tokenizer, block_size, encoding):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.encode = lambda x: self.tokenizer.encode(x, add_special_tokens=True, max_length=block_size+1, padding=True, truncation=True)        
        print(f"Loading Enormous Corpus Start...")
        with open(txt_file, "r", encoding=encoding) as f:
            self.tokens = f.read()
        print(f"Loading Corpus File Done!")

        # self.encoded_token = tokenizer.encode(self.tokens)
        self.length = len(self.tokens) // (self.block_size+1) - 4
        print(f"Dataset Size: {len(self.tokens)}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx + 3) * self.block_size
        # tokens = self.encoded_token[start_idx:end_idx+1]
        t = self.tokens[start_idx: end_idx+1]
        tokens = self.encode(t)
        print(tokens, len(tokens))
        x = torch.tensor(tokens[:-1]).long()
        y = torch.tensor(tokens[1:]).long()
        
        x, y = x.to(device), y.to(device)
        return x, y
    

class ParagraphDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer, block_size):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.blocks = []
        self._load_blocks()

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        block = self.blocks[index]
        encoded_block = self.tokenizer.encode_plus(
            block,
            add_special_tokens=True,
            max_length=self.block_size,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded_block

    def _load_blocks(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            current_block = ""
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    if len(current_block) > 0:
                        blocks = self.tokenizer.batch_encode_plus(
                            [current_block],
                            add_special_tokens=True,
                            max_length=self.block_size,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='pt'
                        )['input_ids']
                        for i in range(blocks.size(1)):
                            self.blocks.append(blocks[:, i:i+self.block_size])
                        current_block = ""
                else:
                    current_block += line + " "
            if len(current_block) > 0:
                blocks = self.tokenizer.batch_encode_plus(
                    [current_block],
                    add_special_tokens=True,
                    max_length=self.block_size,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt'
                )['input_ids']
                for i in range(blocks.size(1)):
                    self.blocks.append(blocks[:, i:i+self.block_size])



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
        
            
        