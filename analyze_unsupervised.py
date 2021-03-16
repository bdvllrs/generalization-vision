import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_results


def load_corr_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    acc = np.load(results_path / "acc.npy", allow_pickle=True).item()
    return acc, config

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
    "CLIP-RN50": ("xkcd:light blue", ""),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", ""),
    "RN50": ("xkcd:orange", ""),
    "BiT-M-R50x1": ("xkcd:hot pink", ""),
    "madry-imagenet_l2_3_0": ("xkcd:light red", ""),
    "madry-imagenet_linf_4": ("xkcd:red", ""),
    "madry-imagenet_linf_8": ("xkcd:dark red", ""),
    "geirhos-resnet50_trained_on_SIN": ("xkcd:light green", ""),
    "geirhos-resnet50_trained_on_SIN_and_IN": ("xkcd:green", ""),
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": ("xkcd:forest green", ""),
}

model_order = ["CLIP-RN50", "virtex", "BiT-M-R50x1", "RN50", "geirhos-resnet50_trained_on_SIN", "geirhos-resnet50_trained_on_SIN_and_IN",
               "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "madry-imagenet_l2_3_0", "madry-imagenet_linf_4",
               "madry-imagenet_linf_8"]

chance_levels = {
    "HouseNumbers": 1/10,
    "CUB": 1/200,
    "CIFAR100": 1/100,
    "MNIST": 1/10,
    "FashionMNIST": 1/10,
    "CIFAR10": 1/10,
}

few_shot_indices = [1, 5, 10]

if __name__ == '__main__':
    n_cols =  1 + len(few_shot_indices)
    n_rows = 1
    figsize = 3

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize * n_cols, figsize * n_rows))

    result_id = 46
    idx_prototypes_bar_plot = 1

    accuracies, config = load_corr_results(Path(f"results/{result_id}"))


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
        if model_name in model_names_short:
            color, hatch = markers[model_name]
            ax.bar([k * 0.35], np.mean(accuracies), 0.35, color=color, hatch=hatch,
                    yerr=(np.std(accuracies) / np.sqrt(len(accuracies))), label=model_names_short[model_name])
    ax.axhline(np.mean(models['chance']), linestyle="--", color="black", label="Average chance level")
    ax.set_title("Clustering")
    # ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_xticks([])
    ax.set_xlabel("")

    print(config)


    result_id = 75
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
            if model_name in model_names_short:
                color, hatch = markers[model_name]
                ax.bar([k * 0.35], np.mean(acc), 0.35, color=color, hatch=hatch,
                       yerr=(np.std(acc) / np.sqrt(len(acc))), label=model_names_short[model_name])

        ax.axhline(np.mean(models['chance']), linestyle="--", color="black", label="Average chance level")
        ax.set_title(f"{few_shot_index}-shot")
        if m == 0:
            ax.set_ylabel("Accuracy")
        ax.set_xticks([])
        ax.set_xlabel("")

    plt.tight_layout(.5)
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"results/{result_id}/averaged_performances.eps", format="eps")
    plt.show()
