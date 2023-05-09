import torch
import torch.optim as optim
import argparse
from torch.utils.data import random_split, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
from transformers import AutoTokenizer
import tqdm
import math
import numpy as np
import time
import logging

from nanoChatGPT import GPT
import utils 
import nanoChatGPT.config as CONFIG
from dataset import TokenedDataset, CoolDataset

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

warmup_iters = 200 # how many steps to warm up for
lr_decay_iters = 6000 
min_lr = 6e-5
# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int, learning_rate: float) -> float:
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


def main(args):
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    learning_rate = args.learning_rate
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    gradient_accumulation_interval = args.gradient_accumulation_interval
    output_dir = args.output_dir
    load = args.load_model
    is_wandb = args.wandb
    with_lr_scheduler = args.with_lr_scheduler
    encoding = args.encoding
    load_mode = args.load_mode
    dataset_path = args.dataset_path
    from_cache = args.from_cache
    save_cache = args.save_cache
    cache_directory = args.cache_directory

    # Using Advanced code for speed up traning.
    is_torch_2 = int(torch.__version__[0]) >= 2

    # Set Up seed.
    utils.set_seed()

    torch.multiprocessing.set_start_method('spawn')
    
    # KoGPT Tokenizer
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    MASK_TOKEN = "[MASK]"

    tokenizer = AutoTokenizer.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16', bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)
    # tokenizer = Tokenizer("./tokenizer/tokenizer.model")

    config = utils.getModelConfig(args.model_size)
    print(args.model_size)

    if is_wandb:
        import wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="nanoChatGPT",
            # track hyperparameters and run metadata
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

    # dataset = GPTDataset(TXT_FILE_PATH, tokenizer, block_size=config.block_size, encoding=encoding)
    # dataset = TokenedDataset(dataset_path, tokenizer=tokenizer, block_size=config.block_size, EOS_TOKEN=EOS_TOKEN, BOS_TOKEN=BOS_TOKEN, load_mode=load_mode, from_cache=from_cache, save_cache=save_cache, cache_destination=cache_directory, device=CONFIG.device, encoding=encoding)
    dataset = CoolDataset(dataset_path, tokenizer, block_size=config.block_size, EOS_TOKEN=EOS_TOKEN, BOS_TOKEN=BOS_TOKEN, device=CONFIG.device)
    total_size = len(dataset)
    train_size = int(0.8*total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=torch.cuda.device_count()*4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=torch.cuda.device_count()*4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
    logger.info("Finishing Loading the Dataset.")

    if load:
        model, optimizer, start_epoch = utils.load_model(output_dir, config, best=True)
    else: 
        os.makedirs(output_dir, exist_ok=True)
        model = GPT(config).to(CONFIG.device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)
        start_epoch = 0

    if is_torch_2:
        model = torch.compile(model)
    
    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        output_dir=output_dir, 
        start_epoch=start_epoch, 
        max_epoch=max_epoch, 
        gradient_accumulation_interval=gradient_accumulation_interval, 
        eval_interval=eval_interval, 
        save_interval=save_interval, 
        learning_rate=learning_rate, 
        with_lr_scheduler=with_lr_scheduler, 
        is_wandb=is_wandb
    )    

def train(model: torch.nn.Module, tokenizer: AutoTokenizer, optimizer: torch.optim.Optimizer, train_loader, val_loader, output_dir: str, start_epoch: int, max_epoch: int, gradient_accumulation_interval: int, eval_interval: int, save_interval: int, learning_rate: float, with_lr_scheduler: bool, is_wandb: bool):   
    scaler = torch.cuda.amp.GradScaler()
    losses = np.zeros(max_epoch)

    for iter in range(start_epoch, start_epoch+max_epoch):
        t0 = time.time()

        # every once in a while evaluate the loss on train and val sets
        # lr = get_lr(iter, learning_rate=learning_rate) if with_lr_scheduler else learning_rate 
        lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Logging the losses.
        _losses = []

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {iter+1}/"+f"{max_epoch+start_epoch}")
        for step, (x, y) in enumerate(pbar):
            # evaluate the loss
            # masked_x = mask_tensor_random_pos(x)
            # _, loss = model(masked_x, y)
            _, loss = model(x, y)
            _losses.append(loss.item())

            scaler.scale(loss).backward()

            if (step+1) % gradient_accumulation_interval == 0 or (step+1) == len(train_loader):
                # Gradient Clipping for Efficient Learning
                # max_norm = 5
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        sample(tokenizer, model)

        dt = time.time() - t0
        mean_loss = utils.mean(_losses)
        losses[iter] = mean_loss
        logger.info(f"Epoch: {iter+1} | Loss: {mean_loss} | Time: {dt*1000:.2f}")

        if (iter-start_epoch) % eval_interval == 0:
            estimated_losses = utils.estimate_loss(model=model, train_loader=train_loader, val_loader=val_loader)
            logger.info(f"EPOCH {iter+1}: train loss {estimated_losses['train']:.4f}, val loss {estimated_losses['val']:.4f}")

        elif is_wandb:
            import wandb
            wandb.log({
                "iter": iter,
                "train/loss": mean_loss,
                "lr": learning_rate
            })

        # Save the every save interval
        if (iter-start_epoch+1) % save_interval == 0:
            utils.save_model(iter+1, model, optimizer, output_dir)

    # finish wandb
    if is_wandb:
        import wandb
        wandb.finish()
    
    return losses

# Generate the sample.
def sample(tokenizer: Tokenizer, model: torch.nn.Module) -> None:
    decode = lambda x: tokenizer.decode(x)
    start_tokens = "[BOS] 세상을 바꾸는 것은 누구일까?"
    result = tokenizer.encode(start_tokens)
    context = torch.tensor(result, device=CONFIG.device, dtype=torch.long)
    context = context.unsqueeze(0)
    result = model.generate(context, max_new_tokens=100)[0].tolist()
    result = decode(result)

    with open('result.txt', "w") as f:
        logger.info(result)
        f.writelines(result)
        f.close()

if __name__ == "__main__":
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

    utils.set_seed()
    
    main(args)
    
