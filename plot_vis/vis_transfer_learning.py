from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import moving_average, markers, model_names_short, dataset_names_short, markers_bars, chance_levels, clip_paper_results
from visiongeneralization.utils import load_results

dataset_order = ["CIFAR10", "CIFAR100", "Caltech101", "DTD", "FGVC-Aircraft", "Food101", "Flowers102",
                 "IIITPets", "SUN397", "StanfordCars", "Birdsnap"]

model_order = ["CLIP-RN50", "virtex", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN",
               "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0",
               "madry-imagenet_linf_4",
               "madry-imagenet_linf_8"]


if __name__ == '__main__':
    # result_id = 212
    result_id = 238
    idx_prototypes_bar_plot = 1

    # config, results_data = load_results(Path(f"../results/{result_id}"))
    # checkpoint = results_data['checkpoint']
    checkpoint = clip_paper_results

    n_rows = 2
    n_cols = 6
    figsize = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    # for k, dataset in enumerate(dataset_order):
    #     i, j = k // n_cols, k % n_cols
    #     for model in checkpoint['train_losses'].keys():
    #         if model in markers:
    #             if dataset in checkpoint['train_losses'][model]:
    #                 y_train = checkpoint['train_losses'][model][dataset]
    #                 y_train = moving_average(y_train, 20)
    #
    #                 color, marker = markers[model]
    #                 ax[i, j].plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
    #     if k == 0:
    #         ax[i, j].legend()
    #     name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
    #     ax[i, j].set_title(name_dataset)
    # plt.tight_layout(pad=.5)
    # plt.show()
    #
    # n_rows = 2
    # n_cols = 6
    # figsize = 3
    # fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))
    #
    # for k, dataset in enumerate(dataset_order):
    #     i, j = k // n_cols, k % n_cols
    #     for model in checkpoint['val_losses'].keys():
    #         if model in markers:
    #             if dataset in checkpoint['val_losses'][model]:
    #                 y_train = checkpoint['val_losses'][model][dataset]
    #                 color, marker = markers[model]
    #                 ax[i, j].plot(y_train, color=color, linestyle=marker, label=model_names_short[model])
    #     if k ==0:
    #         ax[i, j].legend()
    #     name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
    #     ax[i, j].set_title(name_dataset)
    # plt.tight_layout(pad=.5)
    # plt.show()
    #
    # n_rows = 2
    # n_cols = 6
    # figsize = 3
    # fig, ax = plt.subplots(n_rows, n_cols, figsize=(figsize * n_cols, figsize * n_rows))

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
        name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
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
        name_dataset = dataset_names_short[dataset] if dataset in dataset_names_short else dataset
        ax[i, j].set_title(name_dataset)
        ax[i, j].set_xticks([])
        ax[i, j].set_xlabel("")
    plt.tight_layout(pad=.5)
    plt.savefig(f"../results/{result_id}/val_acc_clip_paper.svg", format="svg")
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
    plt.savefig(f"../results/{result_id}/val_acc_summary_clip_paper.svg", format="svg")
    plt.show()

    print(config)
    print(checkpoint)
