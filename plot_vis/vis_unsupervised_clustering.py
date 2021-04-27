from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import markers_bars, dataset_names_short, chance_levels
from visiongeneralization.utils import load_results

model_order = ["CLIP-RN50", "virtex", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN",
               "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0",
               "madry-imagenet_linf_4",
               "madry-imagenet_linf_8", "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"]

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]

few_shot_indices = [1, 5, 10]

if __name__ == '__main__':
    n_cols = 1 + len(few_shot_indices)
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    # result_id = 46
    result_id = 227
    idx_prototypes_bar_plot = 1

    results_data, config = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["accuracies"]

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    ax = axes[-1]

    models = {'chance': []}

    for k, dataset in enumerate(config['datasets']):
        models['chance'].append(chance_levels[dataset['name']])
        for i, (model, model_accuracies) in enumerate(accuracies.items()):
            if dataset['name'] in model_accuracies:
                if model not in models:
                    models[model] = []
                models[model].append(model_accuracies[dataset['name']])

    for k, model_name in enumerate(model_order):
        accuracies = models[model_name]
        if model_name in dataset_names_short:
            color, hatch = markers_bars[model_name]
            ax.bar([k * 0.35], np.mean(accuracies), 0.35, color=color, hatch=hatch,
                   yerr=(np.std(accuracies) / np.sqrt(len(accuracies))), label=dataset_names_short[model_name])
    ax.axhline(np.mean(models['chance']), linestyle="--", color="black", label="Average chance level")
    ax.set_title("Clustering")
    # ax.set_ylabel("Accuracy")
    ax.set_ylim(top=0.8)
    ax.legend()
    ax.set_xticks([])
    ax.set_xlabel("")

    print(config)

    # result_id = 76
    result_id = 229
    idx_prototypes_bar_plot = 1

    accuracies, confusion_matrices, config = load_results(Path(f"results/{result_id}"))

    datasets = {dataset['name']: dataset for dataset in config['datasets']}

    for m, few_shot_index in enumerate(few_shot_indices):
        ax = axes[m]

        models = {'chance': []}

        for k, dataset in enumerate(config['datasets']):
            models['chance'].append(chance_levels[dataset['name']])
            for i, (model, model_accuracies) in enumerate(accuracies.items()):
                if dataset['name'] in model_accuracies:
                    if model not in models:
                        models[model] = []
                    items = sorted(model_accuracies[dataset['name']].items(), key=lambda x: x[0])
                    x, y = zip(*items)
                    mean, std = zip(*y)
                    models[model].append(mean[m])

        for k, model_name in enumerate(model_order):
            acc = models[model_name]
            if model_name in dataset_names_short:
                color, hatch = markers_bars[model_name]
                ax.bar([k * 0.35], np.mean(acc), 0.35, color=color, hatch=hatch,
                       yerr=(np.std(acc) / np.sqrt(len(acc))), label=dataset_names_short[model_name])

        ax.axhline(np.mean(models['chance']), linestyle="--", color="black", label="Average chance level")
        ax.set_title(f"{few_shot_index}-shot")
        if m == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(top=0.8)
        ax.set_xticks([])
        ax.set_xlabel("")

    plt.tight_layout(.5)
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"results/{result_id}/averaged_performances.svg", format="svg")
    plt.show()

    # # Dataset wise plot_vis
    # result_id = 227
    # idx_prototypes_bar_plot = 1
    #
    # accuracies, config = load_corr_results(Path(f"results/{result_id}"))
    #
    # datasets = {dataset['name']: dataset for dataset in config['datasets']}
    #
    # n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    # n_cols = 3
    # n_rows = 2
    # figsize = 3
    #
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))
    #
    # for k, dataset_name in enumerate(dataset_order):
    #     dataset = datasets[dataset_name]
    #     i, j = k // n_cols, k % n_cols
    #     ax = axes[i][j]
    #     for l, (model, model_accuracies) in enumerate(accuracies.items()):
    #         if dataset['name'] in model_accuracies and model in dataset_names_short:
    #             y = model_accuracies[dataset['name']]
    #             color, hatch = markers_bars[model]
    #             ax.bar([l * 0.35], y, 0.35, color=color, hatch=hatch, label=dataset_names_short[model])
    #
    #     ax.axhline(chance_levels[dataset_name], linestyle="--", color="black", label="Chance level")
    #
    #     # if dataset_name == "HouseNumbers":
    #     #     ax.set_ylim(top=0.35)
    #
    #     if k == 0:
    #         ax.legend()
    #     ax.set_xticks([])
    #     ax.set_xlabel("")
    #     if dataset['name'] in dataset_name_plot.keys():
    #         ax.set_title(dataset_name_plot[dataset['name']])
    #     else:
    #         ax.set_title(dataset['name'])
    #     if k == 0:
    #         ax.set_ylabel("Accuracy")
    #     else:
    #         ax.set_ylabel("")
    #         ax.set_xlabel("")
    # # fig.suptitle("Few-shot accuracies on various datasets and models")
    # plt.tight_layout(.5)
    # plt.savefig(f"results/{result_id}/clustering-acc.svg", format="svg")
    # plt.show()
