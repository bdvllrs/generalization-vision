from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import markers_bars, dataset_names_short, model_names_short, chance_levels, plot_config
from visiongeneralization.utils import load_results

model_order = list(reversed([
    "BiT-M-R50x1",
    "geirhos-resnet50_trained_on_SIN",
    "geirhos-resnet50_trained_on_SIN_and_IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    "RN50",
    "madry-imagenet_l2_3_0",
    "madry-imagenet_linf_4",
    "madry-imagenet_linf_8",
    "CLIP-RN50",
    "virtex",
    "GPV-SCE",
    "GPV",
    "TSM-v",
    "ICMLM",
    # "TSM-vat",
    # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
]))

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]

few_shot_indices = [1, 5, 10]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised clustering visualisations')
    parser.add_argument('--load_results', type=int,
                        help='Id of a previous experiment to continue.')
    args = parser.parse_args()
    # Dataset wise plot_vis
    # result_id = 299
    # result_id = 402
    result_id = 449
    # result_id = args.load_results
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["acc"]

    datasets = {dataset['name']: dataset for dataset in config['datasets']}

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    n_cols = 6
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset_name in enumerate(dataset_order):
        dataset = datasets[dataset_name]
        # i, j = k // n_cols, k % n_cols
        # ax = axes[i][j]
        ax = axes[k]
        for l, model in enumerate(model_order):
            model_accuracies = accuracies[model]
            if dataset['name'] in model_accuracies and model in model_names_short:
                y = model_accuracies[dataset['name']]
                color, hatch = markers_bars[model]
                label = model_names_short[model] if k == 0 else None
                ax.bar([l * 0.35], y, 0.35, color=color, hatch=hatch, label=label)

        ax.axhline(chance_levels[dataset_name], linestyle="--", color="black", label=("Chance level" if k == 0 else None))

        # if dataset_name == "HouseNumbers":
        #     ax.set_ylim(top=0.35)

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

        ax.yaxis.label.set_size(plot_config.y_label_font_size)
        ax.xaxis.label.set_size(plot_config.x_label_font_size)
        ax.title.set_size(plot_config.title_font_size)
        ax.tick_params(axis='y', labelsize=plot_config.y_ticks_font_size)
        ax.tick_params(axis='x', labelsize=plot_config.x_ticks_font_size)

    # fig.suptitle("Few-shot accuracies on various datasets and models")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), ncol=7, fontsize=plot_config.legend_font_size)
    plt.tight_layout(pad=.5)
    plt.savefig(f"../results/{result_id}/clustering-acc.svg", format="svg")
    plt.show()
