import glob
import pandas as pd
import json

def create_dataset():
    files = glob.glob("./dataset/NIKLNEWSPAPER_2022_v1.0/*.json")
    result = ''
    for raw_data_path in files:
        with open(raw_data_path) as f:
            js = json.loads(f.read())
        df = pd.DataFrame(js["document"])

        paragraphs = df["paragraph"]
        sentences = [sentence["form"] for article in paragraphs for sentence in article ]

        result += "\n".join(sentences)

    with open("data.txt", "w") as f:
        f.writelines(result)


if __name__ == '__main__':
    create_dataset()
        
            
        