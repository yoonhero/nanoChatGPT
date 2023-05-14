import torch
import torch.optim as optim
import argparse
from torch.utils.data import random_split, DataLoader
import os
import tqdm
import math
import numpy as np
import time
import logging

from nanoChatGPT import GPT
import utils 
import nanoChatGPT.config as CONFIG
from dataset import CoolDataset
from nanoChatGPT.tokenizer import Tokenizer

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s | %(filename)s : %(lineno)s] >> %(message)s')
fileHandler = logging.FileHandler(filename="./training.log")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(level=logging.INFO)

# Train Dataset Optimization with mask the random value of the tensor
def mask_tensor_random_pos(x):
    mask = torch.randn_like(x)>5e-3
    masked_x = torch.where(mask, torch.tensor(0.), x)
    return masked_x

tokenizer = Tokenizer("./tokenizer/corpus.model")

learning_rate = 6e-4
batch_size = 64
micro_batch_size = 5
max_iters = 1000000
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def main(args):
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    gradient_accumulation_interval = args.gradient_accumulation_interval
    output_dir = args.output_dir
    load = args.load_model
    is_wandb = args.wandb
    with_lr_scheduler = args.with_lr_scheduler

    # Using Optimized Torch Code for speed up traning.
    is_torch_2 = int(torch.__version__[0]) >= 2
    
    config = utils.getModelConfig(args.model_size)
    print(args.model_size)

    if is_wandb:
        import wandb
        wandb.init(
            project="nanoChatGPT",
            config={
                "architecture": "GPT",
                "dataset": "Custom Corpus Dataset",
                "max_epoch": max_epoch,
                "block_size": config.block_size,
                "d_model": config.n_embd,
                "n_heads": config.n_heads,
                "n_layer": config.n_layer,
                "vocab": config.vocab_size
            }
        )
        logger.info("Initiate the WANDB.")

    train_loader, val_loader = create_dataloader(args, config)

    if load:
        model, optimizer, _ = utils.load_model(output_dir, config, best=True)
    else: 
        os.makedirs(output_dir, exist_ok=True)
        model = GPT(config).to(CONFIG.device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)

    if is_torch_2:
        model = torch.compile(model)
    
    train(
        model=model,
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        output_dir=output_dir, 
        max_epoch=max_epoch, 
        gradient_accumulation_interval=gradient_accumulation_interval, 
        eval_interval=eval_interval, 
        save_interval=save_interval, 
        learning_rate=learning_rate, 
        with_lr_scheduler=with_lr_scheduler, 
        is_wandb=is_wandb
    )    

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader, val_loader, output_dir: str, max_epoch: int, gradient_accumulation_interval: int, eval_interval: int, save_interval: int, learning_rate: float, with_lr_scheduler: bool, is_wandb: bool):   
    scaler = torch.cuda.amp.GradScaler()
    losses = np.zeros(max_epoch)

    for iter in range(max_epoch):
        t0 = time.time()
        _losses = []
        lr = learning_rate

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {iter+1}/"+f"{max_epoch}")
        for step, (x, y) in enumerate(pbar):
            iter_num = iter * len(train_loader) + step + 1

            if with_lr_scheduler:
                lr = get_lr(iter_num) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            with torch.cuda.amp.autocast():
                _, loss = model(x, y)
                _losses.append(loss.item())

            scaler.scale(loss).backward()

            if (step+1) % gradient_accumulation_interval == 0 or (step+1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if is_wandb:
                    import wandb
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": f"{loss.item():6f}",
                        "lr": lr
                    })
            
            if iter_num % eval_interval == 0:
                estimated_losses = utils.estimate_loss(model=model, train_loader=train_loader, val_loader=val_loader)
                logger.info(f"EPOCH {iter_num}: train loss {estimated_losses['train']:.4f}, val loss {estimated_losses['val']:.4f}")

            # Save the every save interval
            if iter_num % save_interval == 0:
                utils.save_model(iter_num+1, model, optimizer, output_dir)

        dt = time.time() - t0

        sample(model)
        # mean_loss = utils.mean(_losses)
        # losses[iter] = mean_loss
        logger.info(f"Epoch: {iter+1} | Loss: {_losses[-1]} | Time: {dt*1000:.2f}ms")

        if iter_num > max_iters:
            break 

    # finish wandb
    if is_wandb:
        import wandb
        wandb.finish()
    
    return losses

# Generate the sample.
def sample(model: torch.nn.Module) -> None:
    input_texts = "[BOS] 세상을 바꾸는 것은 누구일까?"
    tokens = tokenizer.encode(input_texts, bos=True)
    context = torch.tensor(tokens, device=CONFIG.device, dtype=torch.long)
    context = context.unsqueeze(0)
    result = model.generate(context, max_new_tokens=100)
    result = tokenizer.decode(result[0])

    with open('result.txt', "w") as f:
        logger.info(result)
        f.writelines(result)
        f.close()


def create_dataloader(args, config):
    g = torch.Generator()
    g.manual_seed(12499489)

    batch_size = args.batch_size
    dataset_path = args.dataset_path
    from_cache = args.from_cache
    save_cache = args.save_cache
    cache_directory = args.cache_directory
    
    dataset = CoolDataset(dataset_path, tokenizer, from_cache=from_cache, cache_dir=cache_directory, block_size=config.block_size, device=CONFIG.device, save_cache=save_cache)
    total_size = len(dataset)
    train_size = int(0.8*total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=4, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True,shuffle=False, num_workers=4, generator=g)
    logger.info("Finishing Loading the Dataset.")
    return train_loader, val_loader


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    utils.set_seed()
    torch.set_default_device('cuda')
    # torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Train My Custom GPT 🚀!!!')

    parser.add_argument('--batch_size', type=int, default=CONFIG.batch_size)
    parser.add_argument('--max_epoch', type=int, default=CONFIG.max_epoch)
    parser.add_argument('--learning_rate', type=float, default=CONFIG.learning_rate)
    parser.add_argument('--eval_interval', type=int, default=CONFIG.eval_interval)
    parser.add_argument("--save_interval", type=int, default=CONFIG.save_interval)
    parser.add_argument("--gradient_accumulation_interval", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=CONFIG.TRAINING_OUTPUT_DIR)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--model_size", type=str, default="BASIC")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--with_lr_scheduler", action="store_true")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--load_mode", type=str, default="xml")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--from_cache", action="store_true")
    parser.add_argument("--save_cache", action="store_true")
    parser.add_argument("--cache_directory", type=str)

    args = parser.parse_args()

    main(args)
    
