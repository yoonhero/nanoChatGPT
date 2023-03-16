import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


class CustomTokenizer():
    def __init__(self, tokenizer_path="./vocab/tokenizer"):        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # print(self.tokenizer.encode("안녕").ids)
    
    def encode(self, sentence:str) -> list[int]:
        encoded_str = self.tokenizer.encode(sentence)
        return encoded_str.ids
    
    def decode(self, sentence:str) -> list[int]:
        decoded_str = self.tokenizer.decode(sentence, skip_special_tokens=True)
        return decoded_str

def main(args):
    vocab_size = args.vocab_size
    corpus_file = args.corpus_file

    # Define the tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # Set the pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()

    # Train the tokenizer on a corpus of text
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=5)
    tokenizer.train([corpus_file], trainer)

    tokenizer.save("./vocab/tokenizer")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--vocab_size", type=int, default=10000) 

    args = parser.parse_args()

    # main(args)

    c = CustomTokenizer()
    print(c.decode(c.encode("안녕 난 승현이라고 해")))