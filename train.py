import torch
import torch.optim as optim
import tiktoken
import argparse
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from transformers import AutoTokenizer
import tqdm
import math

from nanoChatGPT import GPTLanguageModel, load_model, save_model, getConfig, batch_size, max_iters, eval_interval, save_interval, learning_rate, device, MODEL_PATH
from dataset import GPTDataset, TokenedDataset


def main(args):
    batch_size = args.batch_size
    max_iters = args.max_iters
    learning_rate = args.learning_rate
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    gradient_accumulation_interval = args.accumulate_interval
    PATH = args.path
    load = args.load_model
    is_wandb = args.wandb
    # max_dataset_size = args.max_dataset_size
    with_lr_scheduler = args.with_lr_scheduler
    encoding = args.encoding

    load_mode = args.load_mode
    dataset_path = args.dataset_path
    from_cache = args.from_cache
    save_cache = args.save_cache
    cache_directory = args.cache_directory

    warmup_iters = 200 # how many steps to warm up for
    lr_decay_iters = 6000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

        

    # KoGPT Tokenizer
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"
    tokenizer = AutoTokenizer.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16', bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)

    config = getConfig(args.model_size)

    if is_wandb:
        import wandb
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

    # dataset = GPTDataset(TXT_FILE_PATH, tokenizer, block_size=config.block_size, encoding=encoding)
    dataset = TokenedDataset(dataset_path, tokenizer=tokenizer, block_size=config.block_size, EOS_TOKEN=EOS_TOKEN, BOS_TOKEN=BOS_TOKEN, load_mode=load_mode, from_cache=from_cache, save_cache=save_cache, cache_destination=cache_directory, device=device, encoding=encoding)
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
        os.makedirs(PATH, exist_ok=True)
        model = GPTLanguageModel(config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.9)
        start_epoch = 0

    # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=max_iters//4, max_iters=max_iters)

    # Train Dataset Optimization with mask the random value of the tensor
    def mask_tensor_random_pos(x):
        mask = torch.randn_like(x)>5e-3
        masked_x = torch.where(mask, torch.tensor(0.), x)
        return masked_x
    
    for iter in range(start_epoch, start_epoch+max_iters):
        # every once in a while evaluate the loss on train and val sets
        lr = get_lr(iter) if with_lr_scheduler else learning_rate 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {iter+1}/"+f"{max_iters+start_epoch}")
        for idx, (x, y) in enumerate(pbar):
            # evaluate the loss
            masked_x = mask_tensor_random_pos(x)
            _, loss = model(masked_x, y)
            losses.append(loss.item())

            loss.backward()

            if (idx+1) % gradient_accumulation_interval == 0 or idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        print(f"Epoch: {iter+1} | Loss: {mean(losses)}")

        if (iter-start_epoch) % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if is_wandb:
                wandb.log({
                    "iter": iter,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr_scheduler.get_lr()[0] if with_lr_scheduler else learning_rate,
                })
        elif is_wandb:
            wandb.log({
                "iter": iter,
                "train/loss": mean(losses),
                "lr": lr_scheduler.get_lr()[0] if with_lr_scheduler else learning_rate,
            })

        # Save the every save interval
        if (iter-start_epoch+1) % save_interval == 0:
            save_model(iter+1, model, optimizer, PATH)


    # finish wandb
    if is_wandb:
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
    parser.add_argument("--accumulate_interval", type=int, default=5)
    parser.add_argument("--path", type=str, default=MODEL_PATH)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--model_size", type=str, default="large")
    parser.add_argument("--wandb", action="store_true")
    # parser.add_argument("--max_dataset_size", type=int, default=1000000)
    parser.add_argument("--with_lr_scheduler", action="store_true")
    parser.add_argument("--encoding", type=str, default="utf-8")

    parser.add_argument("--load_mode", type=str, default="xml")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--from_cache", action="store_true")
    parser.add_argument("--save_cache", action="store_true")
    parser.add_argument("--cache_directory", type=str)

    args = parser.parse_args()

    main(args)
