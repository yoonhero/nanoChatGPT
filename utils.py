import glob 
import torch
from nanoChatGPT.model import GPTLanguageModel
from nanoChatGPT.config import learning_rate, device,LARGE_GPT_CONFIG, SMALL_GPT_CONFIG, KOGPT_CONFIG, LLAMA_7B_CONFIG

# Save the model.
def save_model(epoch: int, model, optimizer, PATH: str) -> None:
    model_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }   

    assert PATH[-1] == "/", "Please check the save directory."
    torch.save(model_state_dict, PATH+f"epoch-{epoch}.tar")

# Load the last training checkpoint result.
def get_last_epoch(PATH: str) -> int:
    """Get the last epoch and TAR file"""
    files = glob.glob(f"{PATH}*")
    if len(files) == 0:
        return None
    
    epochs = [int(filename.split("/")[-1].split(".")[0].split("-")[-1]) for filename in files]
    return max(epochs)

# Load the model with the configuration. 
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


def getModelConfig(model_size):
    configs = {"small":SMALL_GPT_CONFIG, "large":LARGE_GPT_CONFIG, "KOGPT":KOGPT_CONFIG, "LLAMA": LLAMA_7B_CONFIG}
    assert model_size in configs.keys(), "Please Choose Appropriate Model Size"
    config = configs[model_size]

    return config

mean = lambda li: sum(li)/len(li)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        d = train_loader if split=="train" else val_loader

        for X, Y in d:
            _, loss = model(X, Y)
            losses.append(loss.item())

        out[split] = mean(losses)

    model.train()
    return out