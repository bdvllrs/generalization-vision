import argparse
import logging
import os

import gensim.models
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec, PerplexityMetric
from sklearn.decomposition import PCA

from visiongeneralization.utils import load_conf, load_vocab, available_model_names

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_train_val_dataset(conf):
    ntokens_train, ntokens_val = 0, 0
    with open(conf.datasets.enwiki.full, "r", encoding="utf8") as wiki_file:
        sentence_count = len(wiki_file.read().split("\n"))
        train_split = int(0.8 * sentence_count)
        with open(conf.datasets.enwiki.train, "w", encoding="utf8") as train_file:
            for k in range(train_split):
                line = wiki_file.readline()
                ntokens_train += len(line.split(" "))
                train_file.write(line)
        with open(conf.datasets.enwiki.val, "w", encoding="utf8") as val_file:
            for k in range(train_split, sentence_count):
                line = wiki_file.readline()
                ntokens_val += len(line.split(" "))
                val_file.write(line)
    print("ntokens_train", ntokens_train)
    print("ntokens_val", ntokens_val)
    return ntokens_train, ntokens_val


def vocab_trim_rule(word, count, min_count):
    return gensim.utils.RULE_KEEP


class VisualWordEmbeddings:
    def __init__(self, root_path: str, model_name: str, device: torch.device, emd_dimension: int = -1):
        self.root_path = root_path
        self.model_name = model_name
        self.device = device
        self.emb_dimension = emd_dimension

        self.vocabulary = np.load(os.path.join(self.root_path, self.model_name + ".npy"),
                                  allow_pickle=True).item()

        if emb_dimension != -1:
            self.dim_reduction = PCA(n_components=emb_dimension)
            print("Learning projection for PCA...")
            self.dim_reduction.fit(np.vstack(list(self.vocabulary.values())))
            print("done.")

    def get_embedding(self, name):
        if name in self.vocabulary:
            if self.emb_dimension == -1:
                return self.vocabulary[name]
            else:
                return self.dim_reduction.transform(np.expand_dims(self.vocabulary[name], 0))[0]
        return None


class EveryEpochCallback(CallbackAny2Vec):
    def __init__(self, save_dir, model_name, frozen_embeddings, save_models=True, normalize=1.):
        super(EveryEpochCallback, self).__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.losses = []
        self.n_epoch = 0
        self.save_models = save_models
        self.frozen_embeddings = frozen_embeddings
        self.normalize = normalize

    def on_epoch_begin(self, model):
        print(f"Start epoch {self.n_epoch}")
        if self.frozen_embeddings is not None:
            max_diff = 0
            for k, word in enumerate(model.wv.index_to_key):
                if word in self.frozen_embeddings.vocabulary:
                    diff = np.abs(model.wv.vectors[k] - self.frozen_embeddings.get_embedding(word))
                    max_diff = max(max_diff, diff.max())
            print("Diff:", max_diff)

    def on_epoch_end(self, model):
        self.losses.append(model.get_latest_training_loss())
        loss = self.losses[-1] if len(self.losses) == 1 else self.losses[-1] - self.losses[-2]
        print("Loss:", loss / self.normalize)
        if self.save_models:
            model.save(f"{self.save_dir}/{self.model_name.replace('/', '_')}_epoch-{self.n_epoch}.model")
            np.save(f"{self.save_dir}/losses_{self.model_name.replace('/', '_')}.npy", self.losses)
        self.n_epoch += 1


class SavePerplexity(PerplexityMetric):
    def __init__(self):
        super(SavePerplexity, self).__init__()


if __name__ == '__main__':
    conf = load_conf()
    available_models = ["none"]  # baseline skipgram
    available_models.extend(available_model_names(conf))

    parser = argparse.ArgumentParser(description='SkipGram experiment.')
    parser.add_argument('--models', type=str, nargs="+", default=available_models, choices=available_models,
                        help='Model to do.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size.')
    parser.add_argument('--device', default="cuda", type=str,
                        help='Device to use.')
    parser.add_argument('--nepochs', default=5, type=int,
                        help='Number of epochs.')
    parser.add_argument('--min_count', default=12, type=int,
                        help='Min count.')
    parser.add_argument('--window_size', default=5, type=int,
                        help='Window size.')
    parser.add_argument('--vocab_size', default=-1, type=int,
                        help='Size of vocabulary.')
    parser.add_argument('--emb_dimension', default=-1, type=int,
                        help='Size of the embedding.')
    parser.add_argument('--data_location', type=str,
                        help='location to the data information.')
    parser.add_argument('--visual_word_embedding_path', type=str,
                        help='location to the visual word embedding files.')
    parser.add_argument('--save_dir', default=".", type=str,
                        help='location to the save data.')
    parser.add_argument('--load_dir', default=None, type=str,
                        help='path to checkpoints to load.')

    args = parser.parse_args()

    device = torch.device(args.device)
    model_names = args.models
    min_count = args.min_count
    window_size = args.window_size
    emb_dimension = args.emb_dimension
    vocab_size = args.vocab_size

    ntokens_train = 2227749224
    ntokens_val = 372710618

    visual_word_embeddings = None

    for model_name in model_names:
        if model_name != "none":
            print(f"Computing model {model_name}.")
            visual_word_embeddings = VisualWordEmbeddings(args.visual_word_embedding_path,
                                                          model_name, device, emb_dimension)

        if args.load_dir is None:
            # ntokens_train, ntokens_val = split_train_val_dataset(conf)
            visual_words = load_vocab(conf.visual_words)
            vocabulary = load_vocab(conf.vocabulary, vocab_size)

            if emb_dimension == -1:
                emb_dimension = list(visual_word_embeddings.vocabulary.values())[0].shape[0]

            model = Word2Vec(min_count=5, window=5, vector_size=emb_dimension, workers=16, sg=1)

            model.build_vocab([[word] for word in vocabulary], trim_rule=vocab_trim_rule)
            model.build_vocab([[word] for word in visual_words], update=True, trim_rule=vocab_trim_rule)

            if model_name != "none":
                # freeze the vocab of the frozen embeddings.
                model.wv.vectors_lockf = np.ones(model.wv.vectors.shape[0], dtype=np.float32)
                for word in visual_words:
                    if word in model.wv.index_to_key:
                        word_idx = model.wv.get_index(word)
                        model.wv.vectors[word_idx] = visual_word_embeddings.get_embedding(word)
                        model.wv.vectors_lockf[word_idx] = 0.

            print("Start training...")
            model.train(corpus_file=conf.datasets.enwiki.train, total_words=ntokens_train, epochs=args.nepochs,
                        compute_loss=True,
                        callbacks=[
                            EveryEpochCallback(args.save_dir, model_name, visual_word_embeddings,
                                               save_models=(args.load_dir is None),
                                               normalize=ntokens_train)
                        ])

            model.save(f"{args.save_dir}/{model_name.replace('/', '_')}.model")
        else:
            model = Word2Vec.load(os.path.join(args.load_dir, f"{model_name}.model"))

        print(f"Start evaluation {model_name}...")
        model.wv.vectors_lockf = np.zeros(model.wv.vectors.shape[0], dtype=np.float32)

        # for word in frozen_embeddings.vocabulary.keys():
        #     if word in model.wv.index_to_key:
        #         word_idx = model.wv.get_index(word)
        #         print(model.wv.vectors_lockf[word_idx])

        model.train(corpus_file=conf.datasets.enwiki.val, total_words=ntokens_val, epochs=1, compute_loss=True,
                    start_alpha=0, end_alpha=0,
                    callbacks=[
                        EveryEpochCallback(args.save_dir, model_name, visual_word_embeddings, save_models=False,
                                           normalize=ntokens_val)
                    ])
