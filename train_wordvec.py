import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from tqdm import tqdm
from nltk.corpus import wordnet as wn

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

        self.build_vocabulary()

    def build_vocabulary(self):
        for k, classes in enumerate(self.image_net.classes):
            for cls in classes:
                synsets = wn.synsets(cls)
                for synset in synsets:
                    if synset.name() not in self.vocabulary:
                        self.vocabulary[synset.name()] = self.compute_class_embedding(k)

    def compute_class_embedding(self, index):
        return self.class_features[index]

    def get_embedding(self, name):
        synsets = wn.synsets(name)
        for synset in synsets:
            if synset.name() in self.vocabulary.keys():
                return self.vocabulary[synset.name()]
        return None


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12, predefined_embeddings=None):

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        frozen_embeddings = []
        frozen_embedding_id = []
        for word, k in self.data.word2id.items():
            embedding = predefined_embeddings.get_embedding(word)
            if embedding is not None:
                frozen_embeddings.append(embedding)
                frozen_embedding_id.append(k)

        self.frozen_embeddings = torch.stack(frozen_embeddings).to(self.device)
        self.frozen_embedding_id = torch.tensor(frozen_embedding_id).to(self.device)
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension, self.frozen_embeddings, self.frozen_embedding_id)

        self.skip_gram_model.to(self.device)

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    device = torch.device('cuda')
    model, transform = get_model("RN50", device)
    frozen_embeddings = FrozenEmbeddings(model, transform, os.path.expanduser("~/imagenet"), device)

    w2v = Word2VecTrainer("/mnt/HD1/datasets/enwiki/wiki.en.text", "out.vec", predefined_embeddings=frozen_embeddings)
    w2v.train()