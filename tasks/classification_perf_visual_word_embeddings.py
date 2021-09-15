import argparse
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from visiongeneralization.datasets.visual_words_image_net import VisualWordsImagenet
from visiongeneralization.models import get_model
from visiongeneralization.utils import load_conf, available_model_names, evaluate_dataset


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


class TransformImageFeature:
    def __init__(self, dim_reduction):
        self.dim_reduction = dim_reduction

    def __call__(self, features):
        reduced = self.dim_reduction.transform(features.detach().cpu().numpy())
        return torch.from_numpy(reduced).to(features.device)


if __name__ == '__main__':
    conf = load_conf()
    available_models = ["none"]  # baseline skipgram
    available_models.extend(available_model_names(conf, textual=False))

    parser = argparse.ArgumentParser(description='SkipGram experiment.')
    parser.add_argument('--models', type=str, nargs="+", default=available_models, choices=available_models,
                        help='Model to do.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size.')
    parser.add_argument('--emb_dimension', default=300, type=int,
                        help='Size of the embedding.')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = args.models
    emb_dimension = args.emb_dimension


    visual_word_embeddings = None
    accuracies = {}
    confusion_matrices = {}
    losses = {}
    cls_to_idx = {}

    for model_name in model_names:
        if model_name != "none":
            print(f"Computing model {model_name}.")
            visual_word_embeddings = VisualWordEmbeddings(conf.visual_word_embeddings,
                                                          model_name, device, emb_dimension)

            model, transform, _ = get_model(model_name, device)
            model.eval()
            image_net_dataset = VisualWordsImagenet(root=conf.datasets.ImageNet, split='val',
                                                    transform=transform(256, False))
            cls_to_idx[model_name] = image_net_dataset.classes_to_idx

            prototypes = torch.from_numpy(
                np.vstack([visual_word_embeddings.get_embedding(name.lower()) for name in image_net_dataset.classes])
            ).to(device)
            prototypes /= prototypes.norm(dim=-1, keepdim=True)

            accuracy, confusion_matrix = evaluate_dataset(
                model, image_net_dataset, prototypes,
                list(range(len(image_net_dataset.classes))), device,
                encode_text=False,
                batch_size=args.batch_size,
                transform_image_features=TransformImageFeature(visual_word_embeddings.dim_reduction)
            )
            accuracies[model_name] = accuracy
            confusion_matrices[model_name] = confusion_matrix
            print(f"{model_name}, acc: {accuracy}")

    np.save(f"../results/cls_perf_sanity_checks_emb_dim_{emb_dimension}.npy", {
        "accuracies": accuracies,
        "confusion": confusion_matrices,
        "cls_to_idx": cls_to_idx,
        "emb_dimension": emb_dimension
    })
