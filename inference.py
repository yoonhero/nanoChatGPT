import tiktoken
import torch

import utils
from config import PATH, device

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

model, _, _ = utils.load_model(PATH)
model.eval()

with torch.no_grad():
    result = input("")
    while True:
        # generate from the model
        context = torch.tensor(result, dtype=torch.long, device=device)
        result = decode(model.generate(context, max_new_tokens=1)[0].tolist())

        print(f"{result}\n\n")

    