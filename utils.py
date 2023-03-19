import glob 
import torch
from model import GPTLanguageModel
from config import learning_rate, device,LARGE_GPT_CONFIG, SMALL_GPT_CONFIG, KOGPT_CONFIG

def save_model(epoch, model, optimizer, PATH):
    model_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }   
    torch.save(model_state_dict, PATH+f"epoch-{epoch}.tar")

def get_last_epoch(PATH: str) -> int:
    """Get the last epoch and TAR file"""
    files = glob.glob(f"{PATH}*")
    if len(files) == 0:
        return None
    
    epochs = [int(filename.split("/")[-1].split(".")[0].split("-")[-1]) for filename in files]
    return max(epochs)

def load_model(PATH, config, best=True):
    model = GPTLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

    if best:
        assert PATH[-1] == "/", "Please Check the PATH Arguments"
        last_epoch = get_last_epoch(PATH)
        model_state_dict = torch.load(PATH + f"epoch-{last_epoch}.tar")
    else:
        assert PATH[-1] != "/", "Please Check the PATH Arguments"
        model_state_dict = torch.load(PATH)

    model.load_state_dict(model_state_dict["model"])
    optimizer.load_state_dict(model_state_dict["optimizer"])
    start_epoch = model_state_dict["epoch"]

    return model, optimizer, start_epoch


def getConfig(model_size):
    configs = {"small":SMALL_GPT_CONFIG, "large":LARGE_GPT_CONFIG, "KOGPT":KOGPT_CONFIG}
    assert model_size in configs.keys(), "Please Choose Appropriate Model Size"
    config = configs[model_size]

    return config