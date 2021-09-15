from pathlib import Path

import argparse
import matplotlib.pyplot as plt
from math import ceil

import numpy as np

from utils import dataset_names_short, model_names_short, markers, chance_levels, markers_bars, plot_config
from visiongeneralization.utils import load_results

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]
model_order = ["CLIP-RN50",
               "virtex", "ICMLM", "TSM-v", "GPV-SCE", "GPV",
               # "TSM-vat",
               "BiT-M-R50x1", "RN50",
               "geirhos-resnet50_trained_on_SIN",
               "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
               "madry-imagenet_l2_3_0",
               "madry-imagenet_linf_4",
               "madry-imagenet_linf_8",
               # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
               ]

hidden_models = [
    # "CLIP-RN50",
    # "virtex", "ICMLM", "TSM-v",
    # "TSM-vat",
    # "BiT-M-R50x1",
    # "RN50",
    # "geirhos-resnet50_trained_on_SIN",
    # "geirhos-resnet50_trained_on_SIN_and_IN",
    # "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    # "madry-imagenet_l2_3_0",
    # "madry-imagenet_linf_4",
    # "madry-imagenet_linf_8",
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot results visualisations')
    parser.add_argument('--load_results', type=int,
                        help='Id of a previous experiment to continue.')
    args = parser.parse_args()

    # result_id = 448
    result_id = args.load_results
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["accuracies"]

    datasets = {dataset['name']: dataset for dataset in config['datasets']}

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    n_cols = n_datasets
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5))

    for k, dataset_name in enumerate(dataset_order):
        dataset = datasets[dataset_name]
        # i, j = k // n_cols, k % n_cols
        # ax = axes[i, j]
        ax = axes[k]
        for model in model_order:
            model_accuracies = accuracies[model]
            if dataset['name'] in model_accuracies and model in model_names_short:
                items = sorted(model_accuracies[dataset['name']].items(), key=lambda x: x[0])
                x, y = zip(*items)
                y_mean, y_std = zip(*y)
                color, marker = markers[model]
                label = model_names_short[model] if k == 0 else None
                alpha = 0 if model in hidden_models else 1
                ax.errorbar(x, y_mean, y_std, None, color=color, linestyle=marker, label=label, alpha=alpha)

        ax.axhline(chance_levels[dataset_name], linestyle="--", color="black", label=("Chance level" if k == 0 else None))

        ax.set_xticks(x)
        ax.set_xlabel(x)
        if dataset['name'] in dataset_names_short.keys():
            ax.set_title(dataset_names_short[dataset['name']])
        else:
            ax.set_title(dataset['name'])
        if k == 0:
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Number of\nprototypes per class")
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")

        ax.yaxis.label.set_size(plot_config.y_label_font_size)
        ax.xaxis.label.set_size(plot_config.x_label_font_size)
        ax.title.set_size(plot_config.title_font_size)
        ax.tick_params(axis='y', labelsize=plot_config.y_ticks_font_size)
        ax.tick_params(axis='x', labelsize=plot_config.x_ticks_font_size)

    # fig.suptitle("Few-shot accuracies on various datasets and models")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3), ncol=(len(model_order) // 2 + 1), fontsize=plot_config.legend_font_size)
    plt.tight_layout(pad=.5)
    plt.savefig(f"../results/{result_id}/few-shot-acc.svg", format="svg")
    plt.show()

    few_shot_indices = [1, 5, 10]

    n_cols = len(few_shot_indices)
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize * n_cols, figsize * n_rows))

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

        # perfs = [np.mean(models[model_name]) for model_name in model_order if model_name in models]
        # # compute desc order on accuracy
        # perf_order = {order: idx for idx, order in enumerate(list(reversed(np.argsort(perfs))))}

        for k, model_name in enumerate(model_order):
            acc = models[model_name]
            if model_name in model_names_short:
                color, hatch = markers_bars[model_name]
                ax.bar([k * 0.35], np.mean(acc), 0.35, color=color, hatch=hatch,
                       yerr=(np.std(acc) / np.sqrt(len(acc))), label=(model_names_short[model_name] if m == 0 else None))

        ax.axhline(np.mean(models['chance']), linestyle="--", color="black", label=("Average chance level" if m== 0 else None))
        ax.set_title(f"{few_shot_index}-shot")
        if m == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(top=0.8)
        ax.set_xticks([])
        ax.set_xlabel("")

        ax.yaxis.label.set_size(plot_config.y_label_font_size)
        ax.xaxis.label.set_size(plot_config.x_label_font_size)
        ax.title.set_size(plot_config.title_font_size)
        ax.tick_params(axis='y', labelsize=plot_config.y_ticks_font_size)
        ax.tick_params(axis='x', labelsize=plot_config.x_ticks_font_size)

    # fig.subplots_adjust(bottom=0.3, wspace=0.33)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=7, fontsize=plot_config.legend_font_size)
    # fig.legend()
    plt.tight_layout(pad=.5)
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"../results/{result_id}/few_shot_summary.svg", format="svg")
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
