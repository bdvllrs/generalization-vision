from pathlib import Path

import matplotlib.pyplot as plt

from utils import dataset_names_short, model_names_short, markers, chance_levels
from visiongeneralization.utils import load_results

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]
model_order = ["CLIP-RN50", "virtex", "ICMLM", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN",
               "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0",
               "madry-imagenet_linf_4",
               "madry-imagenet_linf_8",
               # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
               ]

if __name__ == '__main__':
    # result_id = 76
    # result_id = 168
    # result_id = 185
    # result_id = 229
    result_id = 345
    # result_id = 291
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["accuracies"]

    datasets = {dataset['name']: dataset for dataset in config['datasets']}

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    n_cols = n_datasets
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset_name in enumerate(dataset_order):
        dataset = datasets[dataset_name]
        # i, j = k // n_cols, k % n_cols
        ax = axes[k]
        for model in model_order:
            model_accuracies = accuracies[model]
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
        if dataset['name'] in dataset_names_short.keys():
            ax.set_title(dataset_names_short[dataset['name']])
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
    plt.savefig(f"../results/{result_id}/few-shot-acc.svg", format="svg")
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
    # plt.savefig(f"../results/{result_id}/plot_bar.svg", format="svg")
    # plt.show()

    print(config)
