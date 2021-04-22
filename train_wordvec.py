import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
import numpy as np

from datasets import RandomizedDataset
from skip_gram_model import SkipGramModel
from utils import get_model, get_set_features, get_prototypes
from wiki_data_utils import DataReader, Word2vecDataset

class FrozenEmbeddings:
    def __init__(self, model, transform, imagenet_path, device, batch_size=32):
        self.model = model
        self.image_net = ImageNet(root=imagenet_path, split='val', transform=transform)

        self.vocabulary = {}

        self.class_features, self.class_features_std, self.class_feature_counts = get_prototypes(
            model, self.image_net, device, n_examples_per_class=-1,
            n_classes=len(self.image_net.classes), batch_size=batch_size
        )

        self.class_features.to("cpu")
        self.class_features_std.to("cpu")
        self.class_feature_counts.to("cpu")

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
            return self.vocabulary[name.lower()]
        return None


class TextFrozenEmbeddings:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.frozen_words = np.load("frozen_words.npy")

    def get_embedding(self, name):
        if name in self.frozen_words:
            return self.model.encode_text(name, self.device)[0]
        return None


class Word2VecTrainer:
    def __init__(self, dataset, output_file, device, emb_dimension=100, batch_size=16, iterations=3,
                 initial_lr=0.001, predefined_embeddings=None):

        self.use_cuda = torch.cuda.is_available()
        self.device = device

        self.data = dataset.data
        # self.data = np.load("word2vec_data.npy", allow_pickle=True).item()

        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        frozen_embeddings = []
        frozen_embedding_id = []
        if predefined_embeddings is not None:
            for word, k in self.data.word2id.items():
                embedding = predefined_embeddings.get_embedding(word)
                if embedding is not None:
                    frozen_embeddings.append(embedding)
                    frozen_embedding_id.append(k)

            self.frozen_embeddings = np.stack(frozen_embeddings)
            pca = PCA(n_components=emb_dimension)

            self.frozen_embeddings = torch.from_numpy(pca.fit_transform(self.frozen_embeddings))
            self.frozen_embedding_id = torch.tensor(frozen_embedding_id)
        else:
            self.frozen_embeddings, self.frozen_embedding_id = None, None
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, self.frozen_embeddings, self.frozen_embedding_id)

        self.skip_gram_model.to(self.device)

    def train(self):

        train_losses = []
        val_losses = []
        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            params = [p for p in self.skip_gram_model.parameters() if p.requires_grad]
            optimizer = optim.SparseAdam(params, lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            val_loss = []
            self.skip_gram_model.train()
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    # train
                    if i < 0.8 * self.data.sentences_count:
                        scheduler.step()
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
                        self.skip_gram_model.eval()
                        with torch.no_grad():
                            loss = self.skip_gram_model(pos_u, pos_v, neg_v)
                            val_loss.append(loss.item())
            val_losses.append(np.mean(val_loss))
            print("Last train BPC:", train_losses[-1])
            print("Val BPC:", val_losses[-1])

            np.save(self.output_file_name.replace(".vec", "_losses.npy"), {
                "train": train_losses,
                "val": val_losses
            })

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    device = torch.device('cuda:2')
    model_names = [
        "BERT",
        "RN50",
        "CLIP-RN50",
        "BiT-M-R50x1",
        "madry-imagenet_l2_3_0",
        "virtex",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "GPT2",
        "madry-imagenet_linf_8",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "madry-imagenet_linf_4",
        "geirhos-resnet50_trained_on_SIN",
    ]
    input_file = "/mnt/SSD/datasets/enwiki/wiki.en.text"
    min_count = 12
    window_size = 5

    data = np.load("word2vec_data.npy", allow_pickle=True).item()
    # data = DataReader(input_file, min_count)
    dataset = Word2vecDataset(data, window_size)
    # dataset = None

    for model_name in model_names:
        print(f"Computing model {model_name}.")
        model, transform = get_model(model_name, device)
        if model_name in ['GPT', 'BERT']:
            frozen_embeddings = TextFrozenEmbeddings(model, device)
        else:
            frozen_embeddings = FrozenEmbeddings(model, transform, "/mnt/SSD/datasets/imagenet", device)

        w2v = Word2VecTrainer(dataset, f"wordvectors/out_{model_name.replace('/', '_')}.vec", device,
                              iterations=5,
                              batch_size=8,
                              emb_dimension=100,
                              predefined_embeddings=frozen_embeddings)
        w2v.train()