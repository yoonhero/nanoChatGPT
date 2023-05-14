import gzip

import ..utils

texts = ""

@utils.profile
def loading():
    global texts
    with gzip.open("./tmp/corpus.txt", 'rb') as f:
        # zipeed_texts = f.read()
        texts = f.read()
    # texts = utils.gunzip_bytes_obj(zipeed_texts)

texts = texts.replace("\n\n===\n\n", "\n")

@utils.profile
def write():
    global texts
    with open("./tmp/corpus.txt", "w") as f:
        f.write(texts)

loading()
write()
