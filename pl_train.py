import pytorch_lightning as pl
import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

import os
import time
import logging
import math
from functools import partial
import tqdm

import utils 
from nanoChatGPT import GPT
import nanoChatGPT.config as CONFIG
from dataset import CoolDataset
from nanoChatGPT.tokenizer import Tokenizer
from nanoChatGPT.model import Block


logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s | %(filename)s : %(lineno)s] >> %(message)s')
fileHandler = logging.FileHandler(filename="./training.log")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(level=logging.INFO)

out_dir = "./tmp/training"
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1
is_wandb = False

compile = False

# Hyperparameters
model_size = "BASIC"
learning_rate = 6e-4
batch_size = 64
micro_batch_size = 5
max_iters = 6000000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5


# Dataset
from_cache = False
save_cache = False
corpus_path = "./tmp/corpus.txt"
cache_dir = "./tmp/cache.csv"
tokenizer_path = "./tokenizer/corpus.model"
tokenizer = Tokenizer(tokenizer_path)


def main():
    devices = 1

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block
    )

    fabric = L.Fabric(
        accelerator="cuda", devices=devices, precision=16, strategy=strategy
    )
    fabric.launch()
    fabric.seed_everything(12499489)

    config = utils.getModelConfig(model_size)

    if is_wandb:
        import wandb
        wandb.init(
            project="nanoChatGPT",
            config={
                "architecture": "GPT",
                "dataset": "AIHUB Corpus Dataset",
                "max_iters": max_iters,
                "block_size": config.block_size,
                "d_model": config.n_embd,
                "n_heads": config.n_heads,
                "n_layer": config.n_layer,
                "vocab": config.vocab_size
            }
        )
        logger.info("Initiate the WANDB.")

    train_loader, val_loader = create_dataloader(config)

    os.makedirs(out_dir, exist_ok=True)

    if val_loader is None:
        train_loader = fabric.setup_dataloaders(train_loader)
    else:
        train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    with fabric.device:
        torch.set_default_dtype(torch.bfloat16)
        model = GPT(config)
        model.apply(model._init_weights)
        torch.set_default_dtype(torch.float32)

    if compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    model, optimizer = fabric.setup(model, optimizer)

    process_batch_size = batch_size // devices
    grad_accum_steps = process_batch_size // micro_batch_size

    train(fabric, model, optimizer, train_loader, val_loader, grad_accum_steps, devices)

    # finish wandb
    if is_wandb:
        import wandb
        wandb.finish()


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    grad_accum_steps: int,
    devices: int,
) -> None:
    step_count = 0

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    for iter_num, (input_ids, targets) in enumerate(tqdm.tqdm(train_dataloader)):
        t0 = time.time()

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            _, loss = model(input_ids, targets)
           
            fabric.backward(loss / grad_accum_steps)

        t1 = time.time()

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dataloader)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                fabric.log_dict(
                    {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
                )

            if step_count % save_interval == 0:
                fabric.print(f"Saving checkpoint to {out_dir}")
                utils.save_model_checkpoint(
                    fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth")
                )

        dt = t1 - t0

        tokens += micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            fabric.log_dict(
                {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
            )
            fabric.print(
                    f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def create_dataloader(config):
    g = torch.Generator()
    g.manual_seed(12499489)

    dataset = CoolDataset(corpus_path, tokenizer, from_cache=from_cache, cache_dir=cache_dir, block_size=config.block_size, save_cache=save_cache, device=CONFIG.device)
    total_size = len(dataset)
    train_size = int(0.8*total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,shuffle=False, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True,shuffle=False,  generator=g)
    logger.info("Finishing Loading the Dataset.")
    return train_loader, val_loader


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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
     # Set Up seed.
    utils.set_seed()
    torch.set_float32_matmul_precision("high")

    main()
    
