import glob 
import torch
from model import GPTLanguageModel
from config import learning_rate, device, S_GPT_CONFIG, LARGE_GPT_CONFIG

def save_model(epoch, model, optimizer, PATH):
    model_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }   
    torch.save(model_state_dict, PATH+f"{epoch}.tar")

def get_last_epoch(PATH: str) -> int:
    """Get the last epoch and TAR file"""
    files = glob.glob(f"{PATH}*")
    if len(files) == 0:
        return None
    
    epochs = [int(filename.split("/")[-1].split(".")[0]) for filename in files]
    return max(epochs)

def load_model(PATH):
    model = GPTLanguageModel(LARGE_GPT_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.9)

    # last_epoch = get_last_epoch(PATH)
    # model_state_dict = torch.load(PATH + f"{last_epoch}.tar")
    model_state_dict = torch.load(PATH)

    model.load_state_dict(model_state_dict["model"])
    optimizer.load_state_dict(model_state_dict["optimizer"])
    start_epoch = model_state_dict["epoch"]

    return model, optimizer, start_epoch