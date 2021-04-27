import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class PlotMarker:
    def __init__(self, n_markers=10):
        self.markers = ["o", "x", ".", "v", "^", "*", "D", "+"]
        # self.colors = ["amber", "brown", "cobalt", "red", "teal", "magenta", "lime", "indigo", "emerald", "mauve", "olive", "orange", "pink"]
        self.colors = ["b", "g", "r", "c", "m", "y", "k"]

        self.possible_markers = [f"{self.markers[k % len(self.markers)]}-{self.colors[k % len(self.colors)]}"
                                 for k in range(n_markers)]

        self.marker_count = 0

    def reset(self):
        self.marker_count = 0

    def get_marker(self):
        marker = self.possible_markers[self.marker_count]
        self.marker_count = (self.marker_count + 1) % len(self.possible_markers)
        return marker


def load_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    checkpoint = np.load(results_path / "checkpoint.npy", allow_pickle=True).item()
    return checkpoint, config


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


model_names_short = {
    "BERT_text": "BERT",
    "GPT2_text": "GPT2",
    # "Word2Vec",
    "CLIP-RN50": "CLIP",
    "CLIP-RN50_text": "CLIP-T",
    "RN50": "RN50",
    "virtex": "VirTex",
    "BiT-M-R50x1": "BiT-M",
    "madry-imagenet_l2_3_0": "AR-L2",
    "madry-imagenet_linf_4": "AR-LI4",
    "madry-imagenet_linf_8": "AR-LI8",
    "geirhos-resnet50_trained_on_SIN": "SIN",
    "geirhos-resnet50_trained_on_SIN_and_IN": "SIN+IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": "SIN+IN-FIN",
}

markers = {
    # "BERT_text": "BERT",
    # "GPT2_text": "GPT2",
    # "Word2Vec",
    "CLIP-RN50": ("xkcd:blue", "-"),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", "--"),
    "RN50": ("xkcd:orange", "-"),
    "BiT-M-R50x1": ("xkcd:puce", "-"),
    "madry-imagenet_l2_3_0": ("xkcd:red", "-"),
    "madry-imagenet_linf_4": ("xkcd:red", "--"),
    "madry-imagenet_linf_8": ("xkcd:red", ":"),
    "geirhos-resnet50_trained_on_SIN": ("xkcd:green", "-"),
    "geirhos-resnet50_trained_on_SIN_and_IN": ("xkcd:green", "--"),
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": ("xkcd:green", ":"),
}

markers_bars = {
    # "BERT_text": "BERT",
    # "GPT2_text": "GPT2",
    # "Word2Vec",
    "CLIP-RN50": ("xkcd:light blue", ""),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", ""),
    "RN50": ("xkcd:orange", ""),
    "BiT-M-R50x1": ("xkcd:puce", ""),
    "madry-imagenet_l2_3_0": ("xkcd:light red", ""),
    "madry-imagenet_linf_4": ("xkcd:red", ""),
    "madry-imagenet_linf_8": ("xkcd:dark red", ""),
    "geirhos-resnet50_trained_on_SIN": ("xkcd:light green", ""),
    "geirhos-resnet50_trained_on_SIN_and_IN": ("xkcd:green", ""),
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": ("xkcd:forest green", ""),
}

dataset_name_plot = {
    "HouseNumbers": "SVHN",
}

dataset_order = ["CIFAR10", "CIFAR100", "Caltech101", "Caltech256", "DTD", "FGVC-Aircraft", "Food101", "Flowers102",
                 "IIITPets", "SUN397", "StanfordCars", "Birdsnap"]
model_order = ["CLIP-RN50", "virtex", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN",
               "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0",
               "madry-imagenet_linf_4",
               "madry-imagenet_linf_8"]

chance_levels = {
    "HouseNumbers": 1 / 10,
    "CUB": 1 / 200,
    "CIFAR100": 1 / 100,
    "MNIST": 1 / 10,
    "FashionMNIST": 1 / 10,
    "CIFAR10": 1 / 10,
    "Caltech101": 1 / 101,
    "Caltech256": 1 / 256,
    "DTD": 1 / 47,
    "FGVC-Aircraft": 1 / 102,
    "Food101": 1 / 101,
    "Flowers102": 1 / 102,
    "IIITPets": 1 / 37,
    "SUN397": 1 / 397,
    "StanfordCars": 1 / 196,
    "Birdsnap": 1 / 500
}

if __name__ == '__main__':
    result_id = 212
    idx_prototypes_bar_plot = 1

    checkpoint, config = load_results(Path(f"results/{result_id}"))

    n_rows = 2
    n_cols = 6
    figsize = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        i, j = k // n_cols, k % n_cols
        for model in checkpoint['train_losses'].keys():
            if model in markers:
                if dataset in checkpoint['train_losses'][model]:
                    y_train = checkpoint['train_losses'][model][dataset]
                    y_train = moving_average(y_train, 20)

                    color, marker = markers[model]
                    ax[i, j].plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
        if k == 0:
            ax[i, j].legend()
        name_dataset = dataset_name_plot[dataset] if dataset in dataset_name_plot else dataset
        ax[i, j].set_title(name_dataset)
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 2
    n_cols = 6
    figsize = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        i, j = k // n_cols, k % n_cols
        for model in checkpoint['val_losses'].keys():
            if model in markers:
                if dataset in checkpoint['val_losses'][model]:
                    y_train = checkpoint['val_losses'][model][dataset]
                    color, marker = markers[model]
                    ax[i, j].plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
        if k ==0:
            ax[i, j].legend()
        name_dataset = dataset_name_plot[dataset] if dataset in dataset_name_plot else dataset
        ax[i, j].set_title(name_dataset)
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 2
    n_cols = 6
    figsize = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        i, j = k // n_cols, k % n_cols
        for model in checkpoint['val_acc'].keys():
            if model in markers:
                if dataset in checkpoint['val_acc'][model]:
                    y_train = checkpoint['val_acc'][model][dataset]
                    color, marker = markers[model]
                    ax[i, j].plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
        ax[i, j].axhline(chance_levels[dataset], linestyle="--", color="black", label="Chance level")
        if k == 0:
            ax[i, j].legend()
        name_dataset = dataset_name_plot[dataset] if dataset in dataset_name_plot else dataset
        ax[i, j].set_title(name_dataset)
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 2
    n_cols = 6
    figsize = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        i, j = k // n_cols, k % n_cols
        n_model = 0
        for model in model_order:
            if model in checkpoint['val_acc'].keys():
                if dataset in checkpoint['val_acc'][model]:
                    y_train = checkpoint['val_acc'][model][dataset]
                    color, hatch = markers_bars[model]
                    ax[i, j].bar([n_model * 0.35], y_train[-1], 0.35, color=color, hatch=hatch,
                                 label=model_names_short[model])
                    n_model += 1
        ax[i, j].axhline(chance_levels[dataset], linestyle="--", color="black", label="Chance level")
        if dataset == "StanfordCars":
            ax[i, j].set_ylim(top=0.1)
        if k == 0:
            ax[i, j].legend()
        name_dataset = dataset_name_plot[dataset] if dataset in dataset_name_plot else dataset
        ax[i, j].set_title(name_dataset)
        ax[i, j].set_xticks([])
        ax[i, j].set_xlabel("")
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 1
    n_cols = 1
    figsize = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    average_accuracy = {}
    for k, dataset in enumerate(dataset_order):
        i, j = k // n_cols, k % n_cols
        n_model = 0
        for model in model_order:
            if model in checkpoint['val_acc'].keys():
                if dataset in checkpoint['val_acc'][model]:
                    y_train = checkpoint['val_acc'][model][dataset]
                    if model not in average_accuracy:
                        average_accuracy[model] = []

                    average_accuracy[model].append(y_train[-1])
    n_model = 0
    for model in model_order:
        if model in average_accuracy:
            color, hatch = markers_bars[model]
            ax.bar([n_model * 0.35], np.mean(average_accuracy[model]), 0.35,
                   yerr=(np.std(average_accuracy[model]) / np.sqrt(len(average_accuracy[model]))), color=color, hatch=hatch,
                   label=model_names_short[model])
            n_model += 1
    ax.axhline(np.mean(list(chance_levels.values())), linestyle="--", color="black", label="Average chance level")
    ax.legend()
    plt.tight_layout(pad=.5)
    ax.set_xticks([])
    ax.set_xlabel("")
    plt.show()

    print(config)
    print(checkpoint)
