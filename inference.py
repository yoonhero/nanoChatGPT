import tiktoken
import torch
import argparse

import utils
from config import MODEL_PATH, device
from tokenizer import CustomTokenizer as Tokenizer

# enc = tiktoken.get_encoding("gpt2")
# encode = lambda s: enc.encode(s)
# decode = lambda l: enc.decode(l)
enc = Tokenizer()
encode = lambda s: enc.encode(s)
decode = lambda s: enc.decode(s)

def main(args):
    model_path = args.path
    max_tokens = args.max_tokens
    start_tokens = args.start
    result = encode(start_tokens)
    model, _, _ = utils.load_model(model_path)
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

        print(f"{result}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train My Custom GPT 🚀!!!')

    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--path", type=str, default=MODEL_PATH)
    parser.add_argument("--start", type=str)

    args = parser.parse_args()

    main(args)
 