import requests
from datasets import load_dataset

# dataset = load_dataset("heegyu/namuwiki-extracted")
# dataset = load_dataset("lcw99/wikipedia-korean-20221001")


urls = {
    "koalphaca":"https://github.com/Beomi/KoAlpaca/blob/main/ko_alpaca_data.json",
}


def download_dataset(url: str, destination: str=""):
    assert destination==""
    
    requests.get(url)


def main():
    for name, url in urls.items(): 
        download_dataset(url, )