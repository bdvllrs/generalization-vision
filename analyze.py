from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import load_results


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

dataset_name_plot = {
    "HouseNumbers": "SVHN"
}

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]

chance_levels = {
    "HouseNumbers": 1/10,
    "CUB": 1/200,
    "CIFAR100": 1/100,
    "MNIST": 1/10,
    "FashionMNIST": 1/10,
    "CIFAR10": 1/10,
}

if __name__ == '__main__':
    result_id = 76
    idx_prototypes_bar_plot = 1

    accuracies, confusion_matrices, config = load_results(Path(f"results/{result_id}"))

    datasets = {dataset['name']: dataset for dataset in config['datasets']}

    plot_marker = PlotMarker(len(config['model_names']))

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    n_cols = n_datasets
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_datasets, figsize))

    for k, dataset_name in enumerate(dataset_order):
        dataset = datasets[dataset_name]
        ax = axes[k]
        for model, model_accuracies in accuracies.items():
            if dataset['name'] in model_accuracies and model in model_names_short:
                items = sorted(model_accuracies[dataset['name']].items(), key=lambda x: x[0])
                x, y = zip(*items)
                y_mean, y_std = zip(*y)
                color, marker = markers[model]
                ax.errorbar(x, y_mean, y_std, None, color=color, linestyle=marker, label=model_names_short[model])

        ax.axhline(chance_levels[dataset_name], linestyle="--", color="black", label="Chance level")

        if k == 0:
            ax.legend()
        ax.set_xticks(x)
        ax.set_xlabel(x)
        if dataset['name'] in dataset_name_plot.keys():
            ax.set_title(dataset_name_plot[dataset['name']])
        else:
            ax.set_title(dataset['name'])
        if k == 0:
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Number of prototypes per class")
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.tight_layout(.5)
    plt.savefig(f"results/{result_id}/few-shot-acc.svg", format="svg")
    plt.show()

    # fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    # gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.4)
    #
    # for k, dataset in enumerate(config['datasets']):
    #     plot_marker.reset()
    #     ax = fig.add_subplot(gs[k // 3, k % 3])
    #     for i, (model, model_accuracies) in enumerate(accuracies.items()):
    #         if dataset['name'] in model_accuracies and model in model_names_short:
    #             items = sorted(model_accuracies[dataset['name']].items(), key=lambda x: x[0])
    #             x, y = zip(*items)
    #             mean, std = zip(*y)
    #             ax.bar([i * 0.35], mean[idx_prototypes_bar_plot], 0.35, yerr=std[idx_prototypes_bar_plot], label=model_names_short[model])
    #
    #     if k == 0:
    #         ax.legend()
    #     ax.set_title(dataset['name'])
    #     ax.set_xticks([])
    #     if k == 0:
    #         ax.set_ylabel("Accuracy")
    #     ax.set_xlabel("")
    #
    # fig.suptitle("5-shot accuracies on various datasets and models")
    # plt.savefig(f"results/{result_id}/plot_bar.svg", format="svg")
    # plt.show()

    print(config)
