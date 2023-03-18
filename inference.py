# import tiktoken
import torch
import argparse

import utils
from utils import getConfig
from config import MODEL_PATH, device
from tokenizer import tokenizer

enc = tokenizer
encode = lambda x: enc.encode(x)
decode = lambda x: enc.decode(x)

def main(args):
    model_path = args.path
    max_tokens = args.max_tokens
    start_tokens = args.start
    result = encode(start_tokens)
    config = utils.getConfig(args.model_size)
    model, _, _ = utils.load_model(model_path, config)
    model.eval()

    if start_tokens == "":
        result = input("")
        result = encode(result)

    with torch.no_grad():
        # for i in range(max_tokens):
        # generate from the model
        context = torch.tensor(result, dtype=torch.long, device=device)
        context = context.unsqueeze(0)
        result = decode(model.generate(context, max_new_tokens=max_tokens)[0].tolist())

        print(f"\n\n{result}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference My Custom GPT 🚀!!!')

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--path", type=str, default=MODEL_PATH)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--model_size", type=str, default="large")

    args = parser.parse_args()

    main(args)
 