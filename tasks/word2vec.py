import argparse
import logging
import os

import gensim.models
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec, PerplexityMetric
from gensim.models.word2vec import LineSentence
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
from torchvision.datasets import ImageNet

from visiongeneralization.models import get_model
from visiongeneralization.text_data_utils import resize_vocabulary
from visiongeneralization.utils import get_set_features

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def vocab_trim_rule(word, count, min_count):
    return gensim.utils.RULE_KEEP


class FrozenEmbeddings:
    def __init__(self, model, transform, imagenet_path, emb_dimension, device, batch_size=32):
        self.model = model
        self.image_net = ImageNet(root=imagenet_path, split='val', transform=transform(256, False))

        self.vocabulary = {}
        self.emb_dimension = emb_dimension

        n_classes = len(self.image_net.classes)
        all_features, labels = get_set_features(model, self.image_net, device, batch_size)
        features = np.zeros((n_classes, all_features.shape[1]))
        std = np.zeros((n_classes, 1))
        counts = np.zeros((n_classes, 1))
        for k in range(n_classes):
            features[k] = np.mean(all_features[labels == k], axis=0)
            std[k] = np.std(all_features[labels == k])
            counts[k] = np.sum(labels == k)
        self.class_features = features
        self.class_features_std = std
        self.class_feature_counts = counts

        if emb_dimension != -1:
            self.dim_reduction = PCA(n_components=emb_dimension)
            print("Learning projection for PCA...")
            self.dim_reduction.fit(all_features)
            print("done.")

        self.build_vocabulary()

    def build_vocabulary(self):
        for k, classes in enumerate(self.image_net.classes):
            cls = classes[0]
            synset = wn.synsets(cls.replace(' ', '_'))[0]
            while len(cls.replace(" ", "_").split("_")) > 1:
                synset = synset.hypernyms()[0]
                cls = synset.lemmas()[0].name()
            if cls.lower() not in self.vocabulary:
                self.vocabulary[cls.lower()] = []
            self.vocabulary[cls.lower()].append(self.compute_class_embedding(k))
        for key, vectors in self.vocabulary.items():
            self.vocabulary[key] = np.vstack(vectors).mean(axis=0).astype(np.float32)

    def compute_class_embedding(self, index):
        return self.class_features[index]

    def get_embedding(self, name):
        if name.lower() in self.vocabulary.keys():
            if self.emb_dimension != -1:
                return self.dim_reduction.transform(np.expand_dims(self.vocabulary[name.lower()], 0))[0]
            else:
                return self.vocabulary[name.lower()]
        return None


class TextFrozenEmbeddings:
    def __init__(self, model, tokenizer, device, emd_dimension=-1):
        self.model = model
        self.device = device
        self.frozen_words = np.load("../frozen_words.npy")
        self.emb_dimension = emd_dimension

        self.vocabulary = {}
        for word in self.frozen_words:
            inputs = tokenizer([f"a photo of a {word}"])
            self.vocabulary[word] = self.model.encode_text(inputs, self.device, 3)[0].detach().cpu().numpy().astype(np.float32)

        if emb_dimension != -1:
            self.dim_reduction = PCA(n_components=emb_dimension)
            print("Learning projection for PCA...")
            self.dim_reduction.fit(np.vstack(list(self.vocabulary.values())))
            print("done.")

    def get_embedding(self, name):
        if name in self.frozen_words:
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
    parser = argparse.ArgumentParser(description='SkipGram experiment.')
    parser.add_argument('--models', type=str, nargs="+", default="RN50", choices=[
        "none",
        "RN50",
        "CLIP-RN50",
        "BiT-M-R50x1",
        "madry-imagenet_l2_3_0",
        "virtex",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "TSM-v",
        "ICMLM",
        "GPT2",
        "BERT",
        "madry-imagenet_linf_8",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "madry-imagenet_linf_4",
        "geirhos-resnet50_trained_on_SIN",
    ],
                        help='Model to do.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size.')
    parser.add_argument('--device', default="cuda", type=str,
                        help='Device to use.')
    parser.add_argument('--nepochs', default=5, type=int,
                        help='Number of epochs.')
    parser.add_argument('--vocab_size', default=-1, type=int,
                        help='Size of vocabulary.')
    parser.add_argument('--emb_dimension', default=-1, type=int,
                        help='Size of the embedding.')
    parser.add_argument('--enwiki_location', type=str,
                        help='Location to the enwiki model.')
    parser.add_argument('--enwiki_val_location', type=str,
                        help='Location to the enwiki model.')
    parser.add_argument('--imagenet_location', type=str,
                        help='location to imagenet dataset.')
    parser.add_argument('--data_location', type=str,
                        help='location to the data information.')
    parser.add_argument('--frozen_words_location', type=str,
                        help='location to the frozen words file.')
    parser.add_argument('--save_dir', default=".", type=str,
                        help='location to the save data.')
    parser.add_argument('--load_dir', default=None, type=str,
                        help='path to checkpoints to load.')

    args = parser.parse_args()
    device = torch.device(args.device)
    model_names = args.models
    input_file = args.enwiki_location
    min_count = 12
    window_size = 5
    emb_dimension = args.emb_dimension
    vocab_size = args.vocab_size

    ntokens_train = 2227749224
    ntokens_val = 372710618
    frozen_embeddings = None

    for model_name in model_names:
        # with open(input_file, "r", encoding="utf8") as wiki_file:
        #     train_split = int(0.8 * data.sentences_count)
        #     with open("/mnt/SSD/datasets/enwiki/wiki.en.train.text", "w", encoding="utf8") as train_file:
        #         for k in range(train_split):
        #             line = wiki_file.readline()
        #             ntokens_train += len(line.split(" "))
        #             train_file.write(line)
        #     with open("/mnt/SSD/datasets/enwiki/wiki.en.val.text", "w", encoding="utf8") as val_file:
        #         for k in range(train_split, data.sentences_count):
        #             line = wiki_file.readline()
        #             ntokens_val += len(line.split(" "))
        #             val_file.write(line)
        # print("ntokens_train", ntokens_train)
        # print("ntokens_val", ntokens_val)

        if model_name != "none":
            print(f"Computing model {model_name}.")
            model, transform, tokenizer = get_model(model_name, device)
            if model_name in ['GPT2', 'BERT']:
                frozen_embeddings = TextFrozenEmbeddings(model, tokenizer, device, emb_dimension)
            else:
                frozen_embeddings = FrozenEmbeddings(model, transform, args.imagenet_location,
                                                     emb_dimension, device)

        if args.load_dir is None:

            data = np.load(args.data_location, allow_pickle=True).item()
            frozen_words = np.load(args.frozen_words_location)

            resize_vocabulary(data, vocab_size)
            val_dataset = LineSentence(args.enwiki_val_location)

            if emb_dimension == -1:
                emb_dimension = list(frozen_embeddings.vocabulary.values())[0].shape[0]

            model = Word2Vec(min_count=5, window=5, vector_size=emb_dimension, workers=16, sg=1)

            model.build_vocab([[word] for word in data.word2id.keys()], trim_rule=vocab_trim_rule)
            model.build_vocab([[word] for word in frozen_words], update=True, trim_rule=vocab_trim_rule)

            if model_name != "none":
                # freeze the vocab of the frozen embeddings.
                model.wv.vectors_lockf = np.ones(model.wv.vectors.shape[0], dtype=np.float32)
                for word in frozen_embeddings.vocabulary.keys():
                    if word in model.wv.index_to_key:
                        word_idx = model.wv.get_index(word)
                        model.wv.vectors[word_idx] = frozen_embeddings.get_embedding(word)
                        model.wv.vectors_lockf[word_idx] = 0.

            print("Start training...")
            model.train(corpus_file=input_file, total_words=ntokens_train, epochs=args.nepochs, compute_loss=True,
                        callbacks=[
                            EveryEpochCallback(args.save_dir, model_name, frozen_embeddings,
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

        model.train(corpus_file=args.enwiki_val_location, total_words=ntokens_val, epochs=1, compute_loss=True,
                    start_alpha=0, end_alpha=0,
                    callbacks=[
                        EveryEpochCallback(args.save_dir, model_name, frozen_embeddings, save_models=False,
                                           normalize=ntokens_val),
                    ])