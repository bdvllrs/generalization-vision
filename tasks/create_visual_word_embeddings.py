import argparse
import os

import numpy as np
import torch
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
from torchvision.datasets import ImageNet

from visiongeneralization.models import get_model
from visiongeneralization.utils import get_set_features, load_conf, load_vocab, available_model_names


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

        with open("visual_words.txt", "w") as f:
            f.write("\n".join(list(self.vocabulary.keys())))

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
    def __init__(self, conf, model, tokenizer, device, emb_dimension=-1):
        self.model = model
        self.device = device
        self.visual_words = load_vocab(conf.visual_words)
        self.emb_dimension = emb_dimension

        self.vocabulary = {}
        for word in self.visual_words:
            inputs = tokenizer([f"a {word}"])
            self.vocabulary[word] = self.model.encode_text(inputs, self.device, 1)[0].detach().cpu().numpy().astype(
                np.float32)

        if emb_dimension != -1:
            self.dim_reduction = PCA(n_components=emb_dimension)
            print("Learning projection for PCA...")
            self.dim_reduction.fit(np.vstack(list(self.vocabulary.values())))
            print("done.")

    def get_embedding(self, name):
        if name in self.visual_words:
            if self.emb_dimension == -1:
                return self.vocabulary[name]
            else:
                return self.dim_reduction.transform(np.expand_dims(self.vocabulary[name], 0))[0]
        return None


if __name__ == '__main__':
    conf = load_conf()

    available_models = ["none"]
    available_models.extend(available_model_names(conf))

    parser = argparse.ArgumentParser(description='Make visual word embedding files.')
    parser.add_argument('--models', type=str, nargs="+", default=available_models, choices=available_models,
                        help='Models to do.')
    parser.add_argument('--device', default="cuda", type=str,
                        help='Device to use.')

    args = parser.parse_args()

    device = torch.device(args.device)
    model_names = args.models

    visual_word_embeddings = None

    for model_name in model_names:
        if model_name != "none":
            print(f"Computing model {model_name}.")
            model, transform, tokenizer = get_model(model_name, device)
            if model_name in ['GPT2', 'BERT']:
                visual_word_embeddings = TextFrozenEmbeddings(conf, model, tokenizer, device, -1)
            else:
                visual_word_embeddings = FrozenEmbeddings(model, transform, conf.datasets.ImageNet,
                                                          -1, device)

        np.save(os.path.join(conf.visual_word_embeddings, f"{model_name}.npy"), visual_word_embeddings.vocabulary)
