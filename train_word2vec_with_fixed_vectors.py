import gensim.models
from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        corpus_path = datapath(self.filepath)
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


if __name__ == '__main__':
    sentences = MyCorpus("/mnt/HD1/datasets/enwiki/wiki.en.text")
    for line in sentences:
        print(line)
        pass
    pass