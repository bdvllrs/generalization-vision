import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import evaluate_dataset, get_prototypes, get_model, get_dataset, RandomizedDataset, save_results, \
    load_results

if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    load_results_id = 7

    # Models to test
    model_names = [
        "CLIP-ViT-B/32",
        "CLIP-RN50",
        "virtex",
        "RN50",
        "BiT-M-R50x1",
        "madry-imagenet_l2_3_0",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
        "geirhos-resnet50_trained_on_SIN",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    ]
    # Dataset to test on
    datasets = [
        {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
        {"name": "CIFAR100", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR10", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
    ]
    # Number of prototypes per class and number of trials for each number of prototype
    prototypes_trials = {n_proto: 10 for n_proto in [1, 5, 10]}

    plot_images = False

    # Make directories to save data
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)

    existing_folders = [int(f.name) for f in results_path.glob("*") if f.is_dir() and f.name.isdigit()]
    result_idx = max(existing_folders) + 1 if len(existing_folders) else 0

    results_path = results_path / str(result_idx)

    config = {
        "model_names": model_names,
        "datasets": datasets,
        "prototypes_trials": prototypes_trials
    }

    if load_results_id is not None:
        accuracies, confusion_matrices, loaded_config = load_results(Path(f"results/{load_results_id}"))
    else:
        accuracies = dict()
        confusion_matrices = dict()

    try:
        with torch.no_grad():
            for model_name in model_names:
                if model_name not in accuracies:
                    accuracies[model_name] = dict()
                if model_name not in confusion_matrices:
                    confusion_matrices[model_name] = dict()
                # Import model
                model, transform = get_model(model_name, device)

                for dataset in datasets:
                    if dataset['name'] not in accuracies[model_name]:
                        accuracies[model_name][dataset['name']] = dict()
                    if dataset['name'] not in confusion_matrices[model_name]:
                        confusion_matrices[model_name][dataset['name']] = dict()

                    # Get dataset
                    dataset_train, dataset_test, class_names = get_dataset(dataset, transform)
                    dataset_train = RandomizedDataset(dataset_train)

                    for n_proto in prototypes_trials.keys():
                        # Average results over several trials
                        trial_accuracies = []
                        trial_confusion_matrices = []
                        n_trials = prototypes_trials[n_proto]

                        if (model_name in accuracies and
                                dataset['name'] in accuracies[model_name] and
                                n_proto in accuracies[model_name][dataset['name']] and
                                model_name in confusion_matrices and
                                dataset['name'] in confusion_matrices[model_name] and
                                n_proto in confusion_matrices[model_name][dataset['name']]):
                            print(
                                f"{model_name}. {dataset['name']}. {n_proto} prototype(s). SKIPPED.")
                            continue

                        for k in range(n_trials):
                            print(
                                f"{model_name}. {dataset['name']}. {n_proto} prototype(s). Trial {k + 1} / {n_trials}.")

                            # Define class prototypes
                            class_features, _, _ = get_prototypes(model, dataset_train.randomize(), device,
                                                            n_examples_per_class=n_proto, n_classes=len(class_names))
                            accuracy, confusion_matrix = evaluate_dataset(model, dataset_test, class_features,
                                                                          list(range(len(class_names))), device,
                                                                          encode_text=False,
                                                                          batch_size=dataset['batch_size'])
                            trial_accuracies.append(accuracy)
                            trial_confusion_matrices.append(confusion_matrix)

                        accuracies[model_name][dataset['name']][n_proto] = (np.mean(trial_accuracies, axis=0),
                                                                            np.std(trial_accuracies, axis=0))
                        confusion_matrices[model_name][dataset['name']][n_proto] = np.mean(confusion_matrix, axis=0)

                        print(f"Accuracy: {accuracy}")
                        if plot_images:
                            plt.imshow(confusion_matrix)
                            plt.title(
                                f"Confusion matrix for {model_name} on {dataset['name']} with {n_proto} prototype(s)")
                            plt.show()

        save_results(results_path, accuracies, confusion_matrices, config)
    except Exception as e:
        print("An error occurred... Saving results so far.")
        save_results(results_path, accuracies, confusion_matrices, config)
        raise e
