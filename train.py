import torch
import torch.optim as optim
import tiktoken
import argparse
from torch.utils.data import Dataset, random_split, DataLoader
import os

from model import GPTLanguageModel
from utils import load_model, save_model
from config import batch_size, max_iters, eval_interval, save_interval, learning_rate, device, MODEL_PATH, TXT_FILE_PATH, load, GPTConfig, S_GPT_CONFIG, LARGE_GPT_CONFIG

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

class GPTDataset(Dataset):
    def __init__(self, txt_file, block_size):
        self.block_size = block_size
        
        # with open(txt_file, "r", encoding="cp949") as f:
        with open(txt_file, "r") as f:
            text = f.read()
        text = text[:1000000]
        self.encoded_texts = encode(text)
        self.length = (len(self.encoded_texts)-block_size) // block_size
        # print(self.length)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.encoded_texts[index*self.block_size:(index+1)*self.block_size]
        y = self.encoded_texts[index*self.block_size+1:(index+1)*self.block_size+1]
        # print(len(x), len(y))
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        x, y = x.to(device), y.to(device)
        return x, y

def main(args):
    batch_size = args.batch_size
    max_iters = args.max_iters
    learning_rate = args.learning_rate
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    PATH = args.path
    TXT_FILE_PATH = args.txt_file_path
    load = args.load_model

    os.makedirs(PATH, exist_ok=True)

    dataset = GPTDataset(TXT_FILE_PATH, block_size=S_GPT_CONFIG.block_size)
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
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.9)
        start_epoch = 0

    print(model)

    for iter in range(start_epoch, start_epoch+max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if iter % save_interval == 0:
            save_model(iter, model, optimizer, PATH)

        for idx, (x, y) in enumerate(train_loader):
            # evaluate the loss
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # scheduler.step()

    # generate from the model``
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    # print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    result = decode(model.generate(context, max_new_tokens=500)[0].tolist())

    with open('result.txt', "w", encoding="cp949") as f:
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
