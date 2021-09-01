from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import moving_average, markers, model_names_short, dataset_names_short, markers_bars, chance_levels, \
    plot_config
from visiongeneralization.utils import load_results

# dataset_order = ["CIFAR10", "CIFAR100", "Caltech101", "DTD", "FGVC-Aircraft", "Food101", "Flowers102",
#                  "IIITPets", "SUN397", "StanfordCars", "Birdsnap"]
dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]

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
    "TSM-v",
    "ICMLM",
    # "TSM-vat",
    # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer learning visualisations')
    parser.add_argument('--load_results', type=int,
                        help='Id of a previous experiment to continue.')
    args = parser.parse_args()
    # result_id = 212
    # result_id = 328
    # result_id = 372
    result_id = 451
    # result_id = args.load_results
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    checkpoint = results_data['checkpoint']
    print(config)
    # _, results_data2 = load_results(Path(f"../results/406"))
    # for item, dic in checkpoint.items():
    #     checkpoint[item]["TSM-vat"]["FashionMNIST"] = results_data2['checkpoint'][item]["TSM-vat"]["FashionMNIST"]
    #
    # save_results(config["results_path"], config, checkpoint=checkpoint)

    # checkpoint = clip_paper_results

    n_rows = 1
    n_cols = 6
    figsize = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        # i, j = k // n_cols, k % n_cols
        ax = axes[k]
        for model in checkpoint['train_losses'].keys():
            if model in markers:
                if dataset in checkpoint['train_losses'][model]:
                    y_train = checkpoint['train_losses'][model][dataset]
                    y_train = moving_average(y_train, 20)

                    color, marker = markers[model]
                    ax.plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
        if k == 0:
            ax.legend()
        name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
        ax.set_title(name_dataset)
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 1
    n_cols = 6
    figsize = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        # i, j = k // n_cols, k % n_cols
        ax = axes[k]
        for model in checkpoint['val_losses'].keys():
            if model in markers:
                if dataset in checkpoint['val_losses'][model]:
                    y_train = checkpoint['val_losses'][model][dataset]
                    color, marker = markers[model]
                    ax.plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
        if k == 0:
            ax.legend()
        name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
        ax.set_title(name_dataset)
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 1
    n_cols = 6
    figsize = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        # i, j = k // n_cols, k % n_cols
        ax = axes[k]
        for model in checkpoint['val_acc'].keys():
            if model in markers:
                if dataset in checkpoint['val_acc'][model]:
                    y_train = checkpoint['val_acc'][model][dataset]
                    color, marker = markers[model]
                    ax.plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
        ax.axhline(chance_levels[dataset], linestyle="--", color="black", label="Chance level")
        if k == 0:
            ax.legend()
        name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
        ax.set_title(name_dataset)
    plt.tight_layout(pad=.5)
    plt.show()

    n_rows = 1
    n_cols = 6
    figsize = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    for k, dataset in enumerate(dataset_order):
        # i, j = k // n_cols, k % n_cols
        # ax = axes[i, j]
        ax = axes[k]
        n_model = 0
        for model in model_order:
            if model in checkpoint['val_acc'].keys():
                if dataset in checkpoint['val_acc'][model]:
                    y_train = checkpoint['val_acc'][model][dataset]
                    color, hatch = markers_bars[model]
                    ax.bar([n_model * 0.35], y_train[-1], 0.35, color=color, hatch=hatch,
                           label=(model_names_short[model] if k == 0 else None))
                    n_model += 1
        ax.axhline(chance_levels[dataset], linestyle="--", color="black", label=("Chance level") if k == 0 else None)
        if dataset == "StanfordCars":
            ax.set_ylim(top=0.1)
        name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
        if k == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(name_dataset)
        ax.set_xticks([])
        ax.set_xlabel("")

        ax.yaxis.label.set_size(plot_config.y_label_font_size)
        ax.xaxis.label.set_size(plot_config.x_label_font_size)
        ax.title.set_size(plot_config.title_font_size)
        ax.tick_params(axis='y', labelsize=plot_config.y_ticks_font_size)
        ax.tick_params(axis='x', labelsize=plot_config.x_ticks_font_size)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=7, fontsize=plot_config.legend_font_size)
    plt.tight_layout(pad=.5)
    plt.savefig(f"../results/{result_id}/transfer_learning_val_acc.svg", format="svg")
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
                   yerr=(np.std(average_accuracy[model]) / np.sqrt(len(average_accuracy[model]))), color=color,
                   hatch=hatch,
                   label=model_names_short[model])
            n_model += 1
    ax.axhline(np.mean(list(chance_levels.values())), linestyle="--", color="black", label="Average chance level")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=7, fontsize=plot_config.legend_font_size)
    plt.tight_layout(pad=.5)
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.yaxis.label.set_size(plot_config.y_label_font_size)
    ax.xaxis.label.set_size(plot_config.x_label_font_size)
    ax.title.set_size(plot_config.title_font_size)
    ax.tick_params(axis='y', labelsize=plot_config.y_ticks_font_size)
    ax.tick_params(axis='x', labelsize=plot_config.x_ticks_font_size)

    plt.savefig(f"../results/{result_id}/transfer_learning_val_acc_summary.svg", format="svg")
    plt.show()

    # print(config)
    # print(checkpoint)
