import json
import re 
import glob
import tqdm
import gzip


def gzip_str(string_: str) -> bytes:
    return gzip.compress(string_.encode())


def gunzip_bytes_obj(bytes_obj: bytes) -> str:
    return gzip.decompress(bytes_obj).decode()


if __name__ == "__main__":
    dirs_to_process = glob.glob("../../dataset/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL1/*/*.json")[:3]

    sources = []

    for filepath in tqdm.tqdm(dirs_to_process):
        with open(filepath, 'r') as file:
            data = json.load(file)

        for entity in data["named_entity"]:
            paragragh = []

            for content in entity["content"]:
                sentence = content["sentence"]
                if "기자 =" in sentence:
                    pos = sentence.index("기자 =") + 5
                    sentence = sentence[pos:]
                elif "관련기사" in sentence or "참조링크" in sentence:
                    sentence = ""    
                if "무단 전재 및 재배포 금지"  in sentence or "무단전재" in sentence or "저작권자ⓒ" in sentence: sentence=""
                
                sentence = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", sentence)
                # sentence = re.sub(r"[a-zA-Z0-9+-_.]@", "", sentence)
                sentence = re.sub(r"^\w+@\(이메일\)\sⓒ\s\(이메일\)", "", sentence)
                sentence = re.sub(r"\([^)]*\)", "", sentence)
                sentence = sentence.replace(",", "")

                paragragh.append(sentence)
                
            sources.append(" ".join(paragragh))
            del paragragh

    result = "\n\n===\n\n".join(sources)

    print(result)

    # gzipped_bytes = gzip_str(result)
    with gzip.open('../dataset/corpus.txt.gz', 'wb') as f:
        f.write(gzip_str(result))  
    

    with gzip.open('../dataset/corpus.txt.gz', 'rb') as f:
        texts = f.read()

        # print(gunzip_bytes_obj(texts))