import glob 
import torch
import torch.nn as nn
from nanoChatGPT.model import GPT
from nanoChatGPT.config import learning_rate, device,LARGE_GPT_CONFIG, SMALL_GPT_CONFIG, KOGPT_CONFIG, LLAMA_7B_CONFIG, GPT_FINAL_CONFIG
from pathlib import Path
import numpy as np
import random
import gzip
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

def save_model_checkpoint(fabric, model, file_path):
    """Handles boilerplate logic for retrieving and saving the state_dict.
    
    This will be upstreamed to Fabric soon.
    """
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

        fabric.save(file_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".pth"))
        return

    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()

    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    fabric.barrier()



# Save the model.
def save_model(epoch: int, model, optimizer, PATH: str) -> None:
    model_state_dict = {
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }   
    save_dir = Path(PATH) / f"epoch-{epoch}.tar"
    torch.save(model_state_dict, str(save_dir))

# Load the last training checkpoint result.
def get_last_epoch(PATH: str) -> int:
    """Get the last epoch and TAR file"""
    path = Path(PATH)
    files = glob.glob(f"{str(path)}/*")
    if len(files) == 0:
        return None
    
    epochs = [int(filename.split("/")[-1].split(".")[0].split("-")[-1]) for filename in files]
    return max(epochs)

# Load the model with the configuration. 
def load_model(PATH, config, best=True):
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))

    if best:
        path = Path(PATH)
        last_epoch = get_last_epoch(str(path))
        path = path / f"epoch-{last_epoch}.tar"
        model_state_dict = torch.load(str(path))
    else:
        path = Path(PATH)
        assert path.exists(), "Please Check the model is existed."
        model_state_dict = torch.load(str(path))

    state_dict = model_state_dict["model"]
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    optimizer.load_state_dict(model_state_dict["optimizer"])
    start_epoch = model_state_dict["epoch"]

    return model, optimizer, start_epoch


def getModelConfig(model_size):
    configs = {"small": SMALL_GPT_CONFIG, "large": LARGE_GPT_CONFIG, "KOGPT": KOGPT_CONFIG, "LLAMA": LLAMA_7B_CONFIG, "BASIC": GPT_FINAL_CONFIG}
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


def gzip_str(string_: str) -> bytes:
    return gzip.compress(string_.encode())


def gunzip_bytes_obj(bytes_obj: bytes) -> str:
    return gzip.decompress(bytes_obj).decode()


def set_seed(seed=12499489):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import cProfile
import io
import pstats
from pstats import SortKey

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper
