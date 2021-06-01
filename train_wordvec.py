import argparse

import numpy as np
import torch
import torch.optim as optim
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm

from skip_gram_model import SkipGramModel
from visiongeneralization.models import get_model
from visiongeneralization.utils import get_set_features
from wiki_data_utils import Word2vecDataset, resize_vocabulary


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
        self.class_features = torch.from_numpy(features).to('cpu')
        self.class_features_std = torch.from_numpy(std).to('cpu')
        self.class_feature_counts = torch.from_numpy(counts).to('cpu')

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
            self.vocabulary[key] = torch.stack(vectors, dim=0).mean(dim=0)

    def compute_class_embedding(self, index):
        return self.class_features[index]

    def get_embedding(self, name):
        if name.lower() in self.vocabulary.keys():
            if self.emb_dimension != -1:
                return self.dim_reduction.transform(self.vocabulary[name.lower()].unsqueeze(0).cpu().numpy())[0]
            else:
                return self.vocabulary[name.lower()].cpu().numpy()
        return None


class TextFrozenEmbeddings:
    def __init__(self, model, tokenizer, device, emd_dimension=-1):
        self.model = model
        self.device = device
        self.frozen_words = np.load("frozen_words.npy")
        self.emb_dimension = emd_dimension

        self.vectors = {}
        for word in self.frozen_words:
            inputs = tokenizer([f"a photo of a {word}"])
            self.vectors[word] = self.model.encode_text(inputs, self.device, 3)[0].detach().cpu().numpy()

        if emb_dimension != -1:
            self.dim_reduction = PCA(n_components=emb_dimension)
            print("Learning projection for PCA...")
            self.dim_reduction.fit(np.vstack(list(self.vectors.values())))
            print("done.")

    def get_embedding(self, name):
        if name in self.frozen_words:
            if self.emb_dimension == -1:
                return self.vectors[name]
            else:
                return self.dim_reduction.transform(np.expand_dims(self.vectors[name], 0))[0]
        return None


class Word2VecTrainer:
    def __init__(self, dataset, output_file, device, emb_dimension=100, batch_size=16, iterations=3,
                 initial_lr=0.001, predefined_embeddings=None):

        self.use_cuda = torch.cuda.is_available()
        self.device = device

        self.data = dataset.data
        # self.data = np.load("word2vec_data.npy", allow_pickle=True).item()

        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=8, pin_memory=True, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        frozen_embeddings = []
        frozen_embedding_id = []
        if predefined_embeddings is not None:
            for word, k in self.data.word2id.items():
                embedding = predefined_embeddings.get_embedding(word)
                if embedding is not None:
                    if self.emb_dimension == -1:
                        self.emb_dimension = embedding.shape[0]
                    frozen_embeddings.append(embedding)
                    frozen_embedding_id.append(k)

            self.frozen_embeddings = torch.from_numpy(np.stack(frozen_embeddings))
            self.frozen_embedding_id = torch.tensor(frozen_embedding_id)
        else:
            self.frozen_embeddings, self.frozen_embedding_id = None, None
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, self.frozen_embeddings,
                                             self.frozen_embedding_id)

        self.skip_gram_model.to(self.device)

    def train(self):
        train_losses = []
        val_losses = []
        params = [p for p in self.skip_gram_model.parameters() if p.requires_grad]
        optimizer = optim.SparseAdam(params, lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for iteration in range(self.iterations):
            print("\n\n\nIteration: " + str(iteration + 1))
            train_items = 0.8 * len(self.dataloader)
            running_loss = 0.0
            val_loss = []
            self.skip_gram_model.train()
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    # train
                    if i < train_items:
                        optimizer.zero_grad()
                        loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                        loss.backward()
                        optimizer.step()

                        train_losses.append(loss.detach().item())
                        running_loss = running_loss * 0.9 + loss.item() * 0.1
                        if i > 0 and i % 500 == 0:
                            print(" Loss: " + str(running_loss))
                    # test
                    else:
                        if i == train_items:
                            self.skip_gram_model.eval()
                            print("EVAL")

                        with torch.no_grad():
                            loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                            val_loss.append(loss.item())
            scheduler.step()
            val_losses.append(np.mean(val_loss))
            print("Last train BPC:", train_losses[-1])
            print("Val BPC:", val_losses[-1])

            np.save(self.output_file_name.replace(".vec", "_losses.npy"), {
                "train": train_losses,
                "val": val_losses
            })

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SkipGram experiment.')
    parser.add_argument('--model', type=str, default="RN50", choices=[
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
    parser.add_argument('--imagenet_location', type=str,
                        help='location to imagenet dataset.')
    parser.add_argument('--data_location', type=str,
                        help='location to the data information.')

    args = parser.parse_args()
    device = torch.device(args.device)
    model_name = args.model
    # input_file = "/mnt/SSD/datasets/enwiki/wiki.en.text"
    input_file = args.enwiki_location
    min_count = 12
    window_size = 5
    emb_dimension = args.emb_dimension
    vocab_size = args.vocab_size

    data = np.load(args.data_location, allow_pickle=True).item()
    # data = DataReader(input_file, min_count, vocab_size)
    resize_vocabulary(data, vocab_size)

    dataset = Word2vecDataset(data, window_size)
    # dataset = None

    print(f"Computing model {model_name}.")
    model, transform, tokenizer = get_model(model_name, device)
    if model_name in ['GPT2', 'BERT']:
        frozen_embeddings = TextFrozenEmbeddings(model, tokenizer, device, emb_dimension)
    else:
        frozen_embeddings = FrozenEmbeddings(model, transform, args.imagenet_location,
                                             emb_dimension,
                                             device)

    w2v = Word2VecTrainer(dataset, f"wordvectors/out_{model_name.replace('/', '_')}.vec", device,
                          iterations=args.nepochs,
                          batch_size=args.batch_size,
                          emb_dimension=emb_dimension,
                          predefined_embeddings=frozen_embeddings)
    w2v.train()
