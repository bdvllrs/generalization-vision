from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import markers_bars, model_names_short, chance_levels, size_training_data
from visiongeneralization.utils import load_results

# model_order = ["CLIP-RN50", "virtex", "ICMLM", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN",
#                "geirhos-resnet50_trained_on_SIN_and_IN",
#                "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0",
#                "madry-imagenet_linf_4",
#                "madry-imagenet_linf_8",
#                # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
#                ]
model_order = list(reversed([
    "CLIP-RN50",
    "madry-imagenet_l2_3_0",
    "madry-imagenet_linf_4",
    "madry-imagenet_linf_8",
    "TSM-v",
    "BiT-M-R50x1",
    "geirhos-resnet50_trained_on_SIN_and_IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    "geirhos-resnet50_trained_on_SIN",
    "RN50",
    "virtex",
    "ICMLM",
    # "TSM-vat",
    # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
]))

dataset_order = ["CIFAR10", "CIFAR100", "CUB", "FashionMNIST", "MNIST", "HouseNumbers"]

few_shot_indices = [1, 5, 10]
figsize = 3

result_id_few_shot = 370
result_id_clustering = 402
result_id_transfer_learning = 372

if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figsize, figsize))
    for k, model_name in enumerate(model_order):
        color, hatch = markers_bars[model_name]
        ax.bar([k * 0.35], size_training_data[model_name], 0.35, color=color, hatch=hatch, log=True,
               label=model_names_short[model_name])
    ax.set_xticks([])
    ax.set_xlabel("")

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=2)
    # fig.legend()
    plt.tight_layout(pad=.5)
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"../results/{result_id_few_shot}/size_training_data.svg", format="svg")
    plt.show()

    n_cols = 2 + len(few_shot_indices)
    n_rows = 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    # result_id = 46
    # result_id = 227
    # result_id = 299
    result_id = result_id_clustering
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["acc"]

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    ax = axes[-2]

    models = {'chance': []}

    for k, dataset in enumerate(config['datasets']):
        models['chance'].append(chance_levels[dataset['name']])
        for i, (model, model_accuracies) in enumerate(accuracies.items()):
            if dataset['name'] in model_accuracies:
                if model not in models:
                    models[model] = []
                models[model].append(model_accuracies[dataset['name']])

    # perfs = [np.mean(models[model_name]) for model_name in model_order]
    # # compute desc order on accuracy
    # perf_order = {order: idx for idx, order in enumerate(list(reversed(np.argsort(perfs))))}
    for k, model_name in enumerate(model_order):
        accuracies = models[model_name]
        if model_name in model_names_short:
            color, hatch = markers_bars[model_name]
            ax.bar([k * 0.35], np.mean(accuracies), 0.35, color=color, hatch=hatch,
                   yerr=(np.std(accuracies) / np.sqrt(len(accuracies))), label=model_names_short[model_name])
    ax.axhline(np.mean(models['chance']), linestyle="--", color="black", label="Average chance level")
    ax.set_title("Unsupervised clustering")
    # ax.set_ylabel("Accuracy")
    ax.set_ylim(top=0.8)
    ax.set_xticks([])
    ax.set_xlabel("")

    print(config)

    result_id = result_id_transfer_learning

    config, results_data = load_results(Path(f"../results/{result_id}"))
    checkpoint = results_data['checkpoint']

    ax = axes[-1]

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
    # perfs = [np.mean(average_accuracy[model_name]) for model_name in model_order if model_name in average_accuracy]
    # # compute desc order on accuracy
    # perf_order = {order: idx for idx, order in enumerate(list(reversed(np.argsort(perfs))))}
    for model in model_order:
        if model in average_accuracy:
            color, hatch = markers_bars[model]
            ax.bar([n_model * 0.35], np.mean(average_accuracy[model]), 0.35,
                   yerr=(np.std(average_accuracy[model]) / np.sqrt(len(average_accuracy[model]))), color=color,
                   hatch=hatch)
            n_model += 1
    ax.axhline(np.mean(list(chance_levels.values())), linestyle="--", color="black")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title("Transfer learning")

    # result_id = 76
    # result_id = 229
    result_id = result_id_few_shot
    # result_id = 291
    idx_prototypes_bar_plot = 1

    config, results_data = load_results(Path(f"../results/{result_id}"))
    accuracies = results_data["accuracies"]
    confusion_matrices = results_data["confusion_matrices"]

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

        # perfs = [np.mean(models[model_name]) for model_name in model_order if model_name in models]
        # # compute desc order on accuracy
        # perf_order = {order: idx for idx, order in enumerate(list(reversed(np.argsort(perfs))))}

        for k, model_name in enumerate(model_order):
            acc = models[model_name]
            if model_name in model_names_short:
                color, hatch = markers_bars[model_name]
                ax.bar([k * 0.35], np.mean(acc), 0.35, color=color, hatch=hatch,
                       yerr=(np.std(acc) / np.sqrt(len(acc))))

        ax.axhline(np.mean(models['chance']), linestyle="--", color="black")
        ax.set_title(f"{few_shot_index}-shot")
        if m == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(top=0.8)
        ax.set_xticks([])
        ax.set_xlabel("")

    # fig.subplots_adjust(bottom=0.3, wspace=0.33)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=7)
    # fig.legend()
    plt.tight_layout(pad=.5)
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"../results/{result_id}/averaged_performances.svg", format="svg")
    plt.show()
