# import tiktoken
import torch
import argparse
from transformers import AutoTokenizer

import utils
from config import MODEL_PATH, device


# KoGPT Tokenizer
enc = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
encode = lambda x: enc.encode(x)
decode = lambda x: enc.decode(x)

@torch.no_grad()
def main(args):
    model_path = args.path
    max_tokens = args.max_tokens
    start_tokens = "[BOS]" + args.start
    result = encode(start_tokens)
    config = utils.getConfig(args.model_size)
    model, _, _ = utils.load_model(model_path, config, best=False)
    model.eval()

    if start_tokens == "":
        result += input("")
        result = encode(result)

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
 