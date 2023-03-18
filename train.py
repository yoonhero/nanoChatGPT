import torch
import torch.optim as optim
import tiktoken
import argparse
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
import pandas as pd
# import wandb

from model import GPTLanguageModel
from utils import load_model, save_model
from config import batch_size, max_iters, eval_interval, save_interval, learning_rate, device, MODEL_PATH, TXT_FILE_PATH, load, GPTConfig, S_GPT_CONFIG, LARGE_GPT_CONFIG, SUPER_SMALL_GPT_CONFIG
from tokenizer import CustomTokenizer as Tokenizer

enc = tiktoken.get_encoding("gpt2")
# enc.special_tokens_set
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
# enc = Tokenizer()
# encode = lambda s: enc.encode(s)
# decode = lambda s: enc.decode(s)

# Data Loading Optimization
class GPTDataset(Dataset):
    def __init__(self, txt_file, block_size):
        self.block_size = block_size
        
        print(f"Loading Enormous Corpus Start...")
        with open(txt_file, "r") as f:
            # self.tokens = f.read()[:1000000].split()
            self.tokens = f.read()[:100000]
        print(f"Loading Corpus File Done!")

        print("Tokenizing...")
        self.encoded_token = encode(self.tokens)
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


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
   

def main(args):
    batch_size = args.batch_size
    max_iters = args.max_iters
    learning_rate = args.learning_rate
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    PATH = args.path
    TXT_FILE_PATH = args.txt_file_path
    load = args.load_model

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="small-chatgpt",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": learning_rate,
    #     "architecture": "GPT",
    #     "dataset": "Custom Corpus Dataset",
    #     "epochs": max_iters,
    #     }
    # )

    os.makedirs(PATH, exist_ok=True)

    dataset = GPTDataset(TXT_FILE_PATH, block_size=LARGE_GPT_CONFIG.block_size)
    total_size = len(dataset)
    train_size = int(0.8*total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = []
            if split == "train":
                for X, Y in train_loader:
                    _, loss = model(X, Y)
                    losses.append(loss.item())
            elif split == "val":
                for X, Y in val_loader:
                    _, loss = model(X, Y)
                    losses.append(loss.item())      
            out[split] = loss.mean()
        model.train()
        return out

    if load:
        model, optimizer, start_epoch = load_model(PATH)
    else: 
        model = GPTLanguageModel(LARGE_GPT_CONFIG).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.9)
        start_epoch = 0

    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=40, max_iters=max_iters)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for iter in range(start_epoch, start_epoch+max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if iter % save_interval == 0:
            save_model(iter, model, optimizer, PATH)

        losses = []
        for idx, (x, y) in enumerate(train_loader):
            # evaluate the loss
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print(f"Loss: {sum(losses)/len(losses)}")
        # wandb.log({"loss": sum(losses)/len(losses)})

    # finish wandb
    # wandb.finish()

    # generate samples
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    result = decode(model.generate(context, max_new_tokens=500)[0].tolist())

    with open('result.txt', "w") as f:
        f.writelines(result)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train My Custom GPT 🚀!!!')

    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--max_iters', type=int, default=max_iters)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--eval_interval', type=int, default=eval_interval)
    parser.add_argument("--save_interval", type=int, default=save_interval)
    parser.add_argument("--path", type=str, default=MODEL_PATH)
    parser.add_argument("--txt_file_path", type=str, default=TXT_FILE_PATH)
    parser.add_argument('--load_model', action='store_true')

    args = parser.parse_args()

    main(args)

    # d = GPTDataset("./dataset/data.txt", 1000)

    # print(d[0])
