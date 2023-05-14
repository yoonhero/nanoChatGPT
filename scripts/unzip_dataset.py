import gzip

import cProfile
import io
import pstats
from pstats import SortKey

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper

texts = ""

@profile
def loading():
    global texts
    with open("./tmp/corpus.txt", 'rb') as f:
        # zipeed_texts = f.read()
        for line in f:
            # texts = f.read()
            if texts not in ["\n", "==="]:
                # texts += line.strip()
                texts += line
    # texts = utils.gunzip_bytes_obj(zipeed_texts)
# texts = texts.replace("\n\n===\n\n", "\n")

@profile
def write():
    global texts
    with open("./tmp/corpus.txt", "w") as f:
        f.write(texts)

loading()
write()
