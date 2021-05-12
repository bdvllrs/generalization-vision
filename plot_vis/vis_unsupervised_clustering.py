from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import markers_bars, dataset_names_short, model_names_short, chance_levels
from visiongeneralization.utils import load_results

model_order = ["CLIP-RN50", "virtex", "ICMLM", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN",
               "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0",
               "madry-imagenet_linf_4",
               "madry-imagenet_linf_8",
               # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
               ]

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]

few_shot_indices = [1, 5, 10]

if __name__ == '__main__':
    # Dataset wise plot_vis
    # result_id = 299
    result_id = 349
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["acc"]

    datasets = {dataset['name']: dataset for dataset in config['datasets']}

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    n_cols = 3
    n_rows = 2
    figsize = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset_name in enumerate(dataset_order):
        dataset = datasets[dataset_name]
        i, j = k // n_cols, k % n_cols
        ax = axes[i][j]
        for l, model in enumerate(model_order):
            model_accuracies = accuracies[model]
            if dataset['name'] in model_accuracies and model in model_names_short:
                y = model_accuracies[dataset['name']]
                color, hatch = markers_bars[model]
                ax.bar([l * 0.35], y, 0.35, color=color, hatch=hatch, label=model_names_short[model])

        ax.axhline(chance_levels[dataset_name], linestyle="--", color="black", label="Chance level")

        # if dataset_name == "HouseNumbers":
        #     ax.set_ylim(top=0.35)

        if k == 0:
            ax.legend()
        ax.set_xticks([])
        ax.set_xlabel("")
        if dataset['name'] in dataset_names_short.keys():
            ax.set_title(dataset_names_short[dataset['name']])
        else:
            ax.set_title(dataset['name'])
        if k == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.tight_layout(.5)
    plt.savefig(f"../results/{result_id}/clustering-acc.svg", format="svg")
    plt.show()
