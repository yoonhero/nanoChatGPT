import gzip

import gzip

def gunzip_bytes_obj(bytes_obj: bytes) -> str:
    return gzip.decompress(bytes_obj).decode()



# @utils.profile
# def loading():
#     global texts
#     with gzip.open("./dataset/corpus.txt.gz", 'rb') as f:
#         zipeed_texts = f.read()
#     texts = utils.gunzip_bytes_obj(zipeed_texts)

# @utils.profile
# def write():
#     with open("./tmp/corpus.txt", "w") as f:
#         f.write(texts)

# loading()
# write()

with gzip.open("./dataset/corpus.txt.gz", 'rb') as f:
    zipeed_texts = f.read()
texts = gunzip_bytes_obj(zipeed_texts)

texts = texts.replace("\n\n===\n\n", "\n")

with open("./tmp/corpus.txt", "w") as f:
    f.write(texts)


