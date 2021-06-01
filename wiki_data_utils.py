import numpy as np
import torch
from tqdm import tqdm


def resize_vocabulary(data, vocab_size=-1):
    id2word = data.id2word.copy()
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
    for idx in most_frequent_words[vocab_size:]:
        freq_unk += word_frequency[idx]

    data.word2id["<unk>"] = vocab_size
    data.id2word[vocab_size] = "<unk>"
    data.word_frequency[vocab_size] = freq_unk

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


class Word2vecDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, "r", encoding="utf8")
        print("Indexing dataset...")
        self.positions = self.index_positions()
        print("done.")

    def index_positions(self):
        positions = []
        for k in tqdm(range(self.data.sentences_count)):
        # for k in tqdm(range(1000)):
            line_position = self.input_file.tell()
            line = self.input_file.readline()
            if len(line):
                words = line.split()
                if len(words) > 1:
                    positions.append(line_position)
        return positions

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx]
        self.input_file.seek(position)
        line = self.input_file.readline()
        words = line.split()
        word_ids = []
        for w in words:
            if w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]:
                word_ids.append(self.data.word2id[w])
            elif w not in self.data.word2id:
                word_ids.append(self.data.word2id['<unk>'])

        boundary = np.random.randint(1, self.window_size)
        result = [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

        return result

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
