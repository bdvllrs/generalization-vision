import numpy as np


def resize_vocabulary(data, vocab_size=-1, additional_vocab=None):
    id2word = data.id2word.copy()
    word2id = data.word2id.copy()
    vocab_size = vocab_size if vocab_size != -1 else len(id2word)
    additional_vocab = [] if additional_vocab is None else additional_vocab
    word_frequency = data.word_frequency.copy()
    data.word_frequency = dict()
    data.word2id = dict()
    data.id2word = dict()
    most_frequent_words, _ = list(zip(*sorted(word_frequency.items(), key=lambda a: a[1], reverse=True)))
    vocabulary = most_frequent_words[:vocab_size]
    freq_unk = 0
    old2newid = {old: k for k, old in enumerate(vocabulary)}
    for idx in vocabulary:
        word = id2word[idx]
        newidx = old2newid[idx]

        data.word2id[word] = newidx
        data.id2word[newidx] = word
        data.word_frequency[newidx] = word_frequency[idx]
    for k, word in enumerate(additional_vocab):
        idx = word2id[word]
        newidx = vocab_size + k

        data.word2id[word] = newidx
        data.id2word[newidx] = word
        data.word_frequency[newidx] = word_frequency[idx]
    # for idx in most_frequent_words[vocab_size:]:
    #     freq_unk += word_frequency[idx]
    #
    # data.word2id["<unk>"] = vocab_size + len(additional_vocab)
    # data.id2word[vocab_size + len(additional_vocab)] = "<unk>"
    # data.word_frequency[vocab_size + len(additional_vocab)] = freq_unk

    data.negatives = []
    data.initTableNegatives()
    data.initTableDiscards()


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count, vocab_size=20_000):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.resize_vocabulary(vocab_size)
        self.initTableNegatives()
        self.initTableDiscards()

    def resize_vocabulary(self, vocab_size):
        id2word = self.id2word.copy()
        word_frequency = self.word_frequency.copy()
        self.word_frequency = dict()
        self.word2id = dict()
        self.id2word = dict()
        most_frequent_words, _ = zip(sorted(word_frequency.items(), key=lambda a, b: b, reverse=True))
        vocabulary = most_frequent_words[:vocab_size]
        freq_unk = 0
        for idx in vocabulary:
            word = id2word[idx]

            self.word2id[word] = idx
            self.id2word[idx] = word
            self.word_frequency[idx] = word_frequency[idx]
        for idx in most_frequent_words[vocab_size:]:
            freq_unk += word_frequency[idx]

        self.word2id["<unk>"] = vocab_size
        self.id2word[vocab_size] = "<unk>"
        self.word_frequency[vocab_size] = freq_unk

    def read_words(self, min_count):
        word_frequency = dict()
        with open(self.inputFileName, encoding="utf8") as f:
            for k, line in enumerate(f):
                line = line.split()
                if len(line) > 1:
                    self.sentences_count += 1
                    for word in line:
                        if len(word) > 0:
                            self.token_count += 1
                            word_frequency[word] = word_frequency.get(word, 0) + 1

                            if self.token_count % 100000000 == 0:
                                print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response
