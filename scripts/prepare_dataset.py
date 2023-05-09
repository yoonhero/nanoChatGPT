import json
import re 
import glob
import tqdm
import gzip
# from transformers import AutoTokenizer
import random

def gzip_str(string_: str) -> bytes:
    return gzip.compress(string_.encode())


def gunzip_bytes_obj(bytes_obj: bytes) -> str:
    return gzip.decompress(bytes_obj).decode()


if __name__ == "__main__":
    dirs_to_process = glob.glob("../../dataset/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL1/*/*.json")

    # dirs_to_process = random.sample(dirs_to_process, 10000)

    print(f"{len(dirs_to_process)} Articles")

    sources = []

    block_size = 256

    for filepath in tqdm.tqdm(dirs_to_process):
        with open(filepath, 'r') as file:
            data = json.load(file)

        for entity in data["named_entity"]:
            paragraph = ""

            for content in entity["content"]:
                if len(paragraph) >= block_size*2:
                    sources.append(paragraph)
                    paragraph = ""

                sentence = content["sentence"]
                if "기자 =" in sentence:
                    pos = sentence.index("기자 =") + 5
                    sentence = sentence[pos:]
                elif "관련기사" in sentence or "참조링크" in sentence:
                    sentence = ""    
                if "무단 전재 및 재배포 금지"  in sentence or "무단전재" in sentence or "저작권자ⓒ" in sentence: sentence=""
                
                sentence = re.sub(r"(\w+)@\(이메일\)\sⓒ\s\(이메일\)", "", sentence)
                sentence = re.sub(r"(\w+)@", "", sentence)
                sentence = re.sub(r"[a-zA-Z0-9+-_.]@[a-zA-Z0-9-]\.[a-zA-Z0-9-.]", "", sentence)
                sentence = re.sub(r"\([^)]*\)", "", sentence)
                sentence = sentence.replace(",", "")

                # paragragh.append(sentence)
                paragraph += sentence
                
            if len(paragraph) >= block_size:
                sources.append(paragraph)

    print(f"total {len(sources)} paragraphs")

    ## -- Test for Tokenize Length --
    # # KoGPT Tokenizer
    # BOS_TOKEN = "[BOS]"
    # EOS_TOKEN = "[EOS]"
    # UNK_TOKEN = "[UNK]"
    # PAD_TOKEN = "[PAD]"
    # MASK_TOKEN = "[MASK]"

    # tokenizer = AutoTokenizer.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16', bos_token=BOS_TOKEN, eos_token=EOS_TOKEN, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)
    # # tokenizer = Tokenizer("./tokenizer/tokenizer.model")

    # for t in sources: 
    #     print(f"Article Length: {len(t)} Tokenized Length: {len(tokenizer.tokenize(t))}")

    result = "\n\n===\n\n".join(sources)

    # gzipped_bytes = gzip_str(result)
    with gzip.open('../dataset/corpus.txt.gz', 'wb') as f:
        f.write(gzip_str(result))  
    

    # with gzip.open('../dataset/corpus.txt.gz', 'rb') as f:
    #     texts = f.read()

        # print(gunzip_bytes_obj(texts))