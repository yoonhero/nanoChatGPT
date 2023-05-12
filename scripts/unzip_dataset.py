import gzip

import utils

texts = ""

@utils.profile
def loading():
    global texts
    with gzip.open("./dataset/corpus.txt.gz", 'rb') as f:
        zipeed_texts = f.read()
    texts = utils.gunzip_bytes_obj(zipeed_texts)

@utils.profile
def write():
    with open("./tmp/corpus.txt", "w") as f:
        f.write(texts)

loading()
write()
