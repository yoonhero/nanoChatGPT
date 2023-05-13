import glob
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import gzip
import numpy as np
from xml.etree.ElementTree import parse
import numpy as np
import tqdm
from nanoChatGPT import device
import time

from nanoChatGPT.tokenizer import Tokenizer

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def merge_dataset(dataset_directories, result_dir:str):
    datasets = glob.glob(f"{dataset_directories}/*.gz")
    assert len(datasets) != 0, "Please check that the dataset file exists."
    result = []

    pbar = tqdm.tqdm(
        datasets,
        smoothing=0,
        leave=True,
        dynamic_ncols=True,
    )
    for dataset_dir in pbar:
        f = gzip.GzipFile(dataset_dir, "r")
        temp_tokens = np.load(f)
        result = np.concatenate((temp_tokens, result), axis=0) if len(result) != 0 else temp_tokens 
    
    with gzip.open(result_dir, "wb") as f:
        np.save(f, result)

def encode_from_texts(texts, tokenizer: Tokenizer, block_size:int, BOS_TOKEN:str, EOS_TOKEN:str):
    tokens = []
    pbar = tqdm.tqdm(
        texts,
        smoothing=0,
        leave=True,
        dynamic_ncols=True,
    )
    for text in pbar:
        # print(text)
        if text == "":
            continue
        
        # text = text.replace("\n", f"{SEP_TOKEN}")
        text = f"{BOS_TOKEN} {text} {EOS_TOKEN}" 

        temp_tokens = np.array(tokenizer.encode(text), dtype=np.int64)
        length = len(temp_tokens)
        padding = -length % (block_size+1)
        temp_tokens = np.reshape(np.concatenate((temp_tokens, np.ones(padding)*tokenizer.encode("[PAD]"))), (-1, block_size+1))
        # print(temp_tokens.shape)
        tokens = np.concatenate((tokens, temp_tokens), axis=0) if len(tokens) != 0 else temp_tokens

    return tokens

def encode_from_texts_v2(texts, tokenizer, block_size:int):
    tokens = []
    pbar = tqdm.tqdm(
        texts,
        smoothing=0,
        leave=True,
        dynamic_ncols=True,
    )
    for text in pbar:
        # print(text)
        if text == "":
            continue
        
        encoded_text = tokenizer.encode(text, bos=True, eos=True)
        temp_tokens = np.array(encoded_text, dtype=np.int64)
        length = len(temp_tokens)
        padding = -length % (block_size+1)
        temp_tokens = np.reshape(np.concatenate((temp_tokens, np.ones(padding)*tokenizer.pad_id)), (-1, block_size+1))
        tokens = np.concatenate((tokens, temp_tokens), axis=0) if len(tokens) != 0 else temp_tokens

    return tokens

def read_text_from_xml(xml_dir:str):
    try:
        tree = parse(xml_dir)
        root = tree.getroot()
        text = " ".join([x.text for x in root.findall("text")[0].findall("p")])
        return text
    except: return ''

def encode_text_from_xml(folder_dir: str, tokenizer: Tokenizer, block_size:int, BOS_TOKEN:str, EOS_TOKEN:str):
    assert folder_dir[-1] != "/", "Check the directory please."
    xml_file_directories = glob.glob(f"{folder_dir}/*")

    texts = [read_text_from_xml(xml_dir) for xml_dir in xml_file_directories]
    
    tokens = encode_from_texts(texts, tokenizer, block_size, BOS_TOKEN, EOS_TOKEN)

    return tokens

def read_text_from_txt(txt_dir: str, encoding):
    with open(txt_dir, "r", encoding=encoding) as f:
        texts = f.read()
    return texts

def encode_text_from_txt(folder_dir: str, tokenizer: Tokenizer, block_size: int, encoding):
    assert folder_dir[-1] != "/", "Check the directory please."
    txt_file_directories = glob.glob(f"{folder_dir}/*")

    texts = [read_text_from_txt(txt_dir) for txt_dir in txt_file_directories]
    
    tokens = encode_from_texts(texts, tokenizer, block_size)

    return tokens

class CoolDataset(Dataset):
    def __init__(
            self, 
            corpus_path:str, 
            tokenizer:Tokenizer, 
            from_cache:bool,
            cache_dir: str,
            block_size:int, 
            EOS_TOKEN:str,
            BOS_TOKEN:str,
            device="cuda",
        ):
        self.device = device
        self.block_size = block_size
        self.EOS_TOKEN = EOS_TOKEN
        self.BOS_TOKEN = BOS_TOKEN

        self.tokenizer = tokenizer

        self.cache_dir = cache_dir
        self.pool_size = 4

        tokens = []
        if not from_cache:
            start = time.time()
            with open(corpus_path, "r", buffering=100000) as f:
                print("Loading Corpus Line by Line and Tokenizing")
                for line in tqdm.tqdm(f):
                    token = self.tokenizer.encode(line, bos=True, eos=True, max_length=self.block_size+1, pad=True)
                    tokens.append(token)

            print(f"Loading Done in {time.time() - start:.4f}s")

            self.num_subsets = len(tokens)
            self.tokens = np.array(tokens, dtype=np.int64)
            del tokens
            self.save_cache(self.cache_dir)

        else:
            f = gzip.GzipFile(self.cache_dir, "r")
            self.tokens = np.load(f)
            self.num_subsets = self.tokens.shape[0]
            return

    def __len__(self):
        return self.num_subsets

    def save_cache(self, cache_destination):
        with gzip.open(cache_destination, "wb") as f:
            np.save(f, self.tokens)

    def __getitem__(self, idx): 
        token = self.tokens[idx]
        x = torch.as_tensor(token[:-1], dtype=torch.long, device=self.device)
        y = torch.as_tensor(token[1:], dtype=torch.long, device=self.device)

        return x, y

    def __repr__(self) -> str: 
        return f"CoolDataset containing {self.num_subsets} subsets."
    

class TokenedDataset(Dataset):
    def __init__(
            self, 
            file_path:str, 
            tokenizer:Tokenizer, 
            block_size:int, 
            EOS_TOKEN:str,
            BOS_TOKEN:str,
            load_mode: str="xml",
            from_cache:bool=False, 
            save_cache: bool=False, 
            cache_destination: str = "dataset_cache.tar.gz",
            device="cuda",
            encoding: str="utf-8"
        ):
        self.device = device
        self.block_size = block_size

        if from_cache:
            # open_func = gzip.open if file_path.endswith(".gz") else open
            # with open_func(cache_destination, "rb") as f:
            #     self.tokens = np.load(f, allow_pickle=True)
            f = gzip.GzipFile(cache_destination, "r")
            self.tokens = np.load(f)
            self.num_subsets = self.tokens.shape[0]
            return
        
        mode = ["xml", "txt", "csv"]
        assert load_mode in mode, "Please Select Appropriate Mode for Dataset Loading."
        if load_mode=="xml":
            self.tokens = encode_text_from_xml(file_path, tokenizer=tokenizer, block_size=block_size, EOS_TOKEN=EOS_TOKEN, BOS_TOKEN=BOS_TOKEN)
            self.num_subsets = self.tokens.shape[0]
        elif load_mode=="txt":
            self.tokens = encode_text_from_txt(file_path, tokenizer=tokenizer, block_size=block_size, encoding=encoding, EOS_TOKEN=EOS_TOKEN, BOS_TOKEN=BOS_TOKEN)
            self.num_subsets = self.tokens.shape[0]

        if save_cache:
            self.save_cache(cache_destination)

    def save_cache(self, cache_destination):
        with gzip.open(cache_destination, "wb") as f:
            np.save(f, self.tokens)

    def __len__(self):
        return self.num_subsets

    def __getitem__(self, idx):
        x = torch.as_tensor(self.tokens[idx][:-1], dtype=torch.long, device=self.device)
        y = torch.as_tensor(self.tokens[idx][1:], dtype=torch.long, device=self.device)

        return x, y

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
        # print(tokens, len(tokens))
        x = torch.tensor(tokens[:-1]).long()
        y = torch.tensor(tokens[1:]).long()
        
        x, y = x.to(device), y.to(device)
        return x, y
    
def old_create_dataset():
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
    # create_dataset()
    tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )
    # encode_text_from_xml("./dataset/NIKL_NP_v1.2/malmungchi", tokenizer=tokenizer, block_size=128)
        
    dataset = TokenedDataset("./dataset/NIKL_NP_v1.2/malmungchi", tokenizer=tokenizer, block_size=128, save_cache=True, BOS_TOKEN="[BOS]", EOS_TOKEN="[EOS]")
    print(dataset[0])
    print(dataset[10])
            
        