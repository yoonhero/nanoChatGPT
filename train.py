import torch
import torch.optim as optim
import tiktoken
import argparse
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
import pandas as pd
import wandb
from transformers import AutoTokenizer
import tqdm

from model import GPTLanguageModel
from utils import load_model, save_model,getConfig
from dataset import GPTDataset
from config import batch_size, max_iters, eval_interval, save_interval, learning_rate, device, MODEL_PATH, TXT_FILE_PATH


# enc = tiktoken.get_encoding("gpt2")
# encode = lambda s: enc.encode(s)
# decode = lambda l: enc.decode(l)
# enc = Tokenizer()
# encode = lambda s: enc.encode(s)
# decode = lambda s: enc.decode(s)

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
    is_wandb = args.wandb
    # max_dataset_size = args.max_dataset_size
    with_lr_scheduler = args.with_lr_scheduler

    # KoGPT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
    )

    config = getConfig(args.model_size)

    if is_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="small-chatgpt",
            
            # track hyperparameters and run metadata
            config={
            "architecture": "GPT",
            "dataset": "Custom Corpus Dataset",
            "epochs": max_iters,
            "block_size": config.block_size,
            "d_model": config.n_embd,
            "n_heads": config.n_heads,
            "n_layer": config.n_layer,
            "vocab": config.vocab_size
            }
        )

    os.makedirs(PATH, exist_ok=True)

    dataset = GPTDataset(TXT_FILE_PATH, tokenizer, block_size=config.block_size)
    total_size = len(dataset)
    train_size = int(0.8*total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    mean = lambda li: sum(li)/len(li)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = []
            if split == "train":
                d = train_loader
            elif split == "val":
                d = val_loader
            
            for X, Y in d:
                _, loss = model(X, Y)
                losses.append(loss.item())

            out[split] = mean(losses)
        model.train()
        return out

    if load:
        model, optimizer, start_epoch = load_model(PATH, config)
    else: 
        model = GPTLanguageModel(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.9)
        start_epoch = 0

    if with_lr_scheduler:
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=max_iters//4, max_iters=max_iters)

    for iter in range(start_epoch, start_epoch+max_iters):
        # every once in a while evaluate the loss on train and val sets
        if (iter-start_epoch) % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if is_wandb:
                wandb.log({
                    "iter": iter,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr_scheduler.get_lr()[0],
                })
        if (iter-start_epoch+1) % save_interval == 0:
            save_model(iter+1, model, optimizer, PATH)

        losses = []
        for idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {iter+1}/"+f"{max_iters+start_epoch}")):
            # evaluate the loss
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if with_lr_scheduler:
                lr_scheduler.step()

        print(f"Epoch: {iter} | Loss: {mean(losses)}")

    # finish wandb
    if wandb:
        wandb.finish()

    # generate samples
    decode = lambda x: tokenizer.decode(x)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
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
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--wandb", action="store_true")
    # parser.add_argument("--max_dataset_size", type=int, default=1000000)
    parser.add_argument("--with_lr_scheduler", action="store_true")

    args = parser.parse_args()

    main(args)
