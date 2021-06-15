import argparse
import os

import numpy as np
import torch

from visiongeneralization.datasets.datasets import get_dataset, RandomizedDataset
from visiongeneralization.models import get_model
from visiongeneralization.utils import evaluate_dataset, get_prototypes, run, save_results, load_conf


def get_args():
    parser = argparse.ArgumentParser(description='Few-shot generalization task.')
    parser.add_argument('--load_results', default=None, type=int,
                        help='Id of a previous experiment to continue.')
    parser.add_argument('--ntrials', default=10, type=int,
                        help='Number of trials to execute.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size.')

    return parser.parse_args()


def main(config, accuracies, confusion_matrices):
    try:
        model_names, datasets = config["model_names"], config["datasets"]
        prototypes_trials = config["prototypes_trials"]
        override_models = config["override_models"]

        # Remove some modes to recompute them.
        for model in override_models:
            del accuracies[model]
            del confusion_matrices[model]

        with torch.no_grad():
            for model_name in model_names:
                if model_name not in accuracies:
                    accuracies[model_name] = dict()
                if model_name not in confusion_matrices:
                    confusion_matrices[model_name] = dict()

                # Import model
                model, transform, _ = get_model(model_name, device)
                model.eval()

                for dataset in datasets:
                    # initialize results dicts
                    if dataset['name'] not in accuracies[model_name]:
                        accuracies[model_name][dataset['name']] = dict()
                    if dataset['name'] not in confusion_matrices[model_name]:
                        confusion_matrices[model_name][dataset['name']] = dict()

                    # Get dataset
                    dataset_train, dataset_test, class_names, _ = get_dataset(dataset, transform)
                    dataset_train = RandomizedDataset(dataset_train)

                    # iterate over number of shots (1, 5, 10 shots)
                    for n_proto in prototypes_trials.keys():
                        trial_accuracies = []
                        trial_confusion_matrices = []
                        n_trials = prototypes_trials[n_proto]

                        # check that data has not been already computed
                        if (model_name in accuracies and
                                dataset['name'] in accuracies[model_name] and
                                n_proto in accuracies[model_name][dataset['name']] and
                                model_name in confusion_matrices and
                                dataset['name'] in confusion_matrices[model_name] and
                                n_proto in confusion_matrices[model_name][dataset['name']]):
                            print(
                                f"{model_name}. {dataset['name']}. {n_proto} prototype(s). SKIPPED.")
                            continue

                        # do several iterations per experiment
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

                        print(f"Accuracy: {np.mean(trial_accuracies, axis=0)}")

        save_results(config["results_path"], config,
                     accuracies=accuracies, confusion_matrices=confusion_matrices)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(config["results_path"], config,
                     accuracies=accuracies, confusion_matrices=confusion_matrices)
        raise e


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = get_args()
    conf = load_conf()

    load_results_id = args.load_results
    batch_size = args.batch_size

    # Models to test
    model_names = [
        "TSM-v",
        "TSM-vat",
        "ICMLM",
        # "semi-supervised-YFCC100M",
        # "semi-weakly-supervised-instagram",
        "geirhos-resnet50_trained_on_SIN",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "madry-imagenet_l2_3_0",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
        "CLIP-ViT-B/32",
        "CLIP-RN50",
        "virtex",
        "RN50",
        "BiT-M-R50x1",
    ]

    # Dataset to test on
    datasets = [
        {"name": "CIFAR10", "batch_size": args.batch_size, "root_dir": conf.datasets.CIFAR10},
        {"name": "HouseNumbers", "batch_size": args.batch_size, "root_dir": conf.datasets.SVHN},
        {"name": "CUB", "batch_size": args.batch_size, "root_dir": conf.datasets.CUB},
        {"name": "CIFAR100", "batch_size": args.batch_size, "root_dir": conf.datasets.CIFAR100},
        {"name": "MNIST", "batch_size": args.batch_size, "root_dir": conf.datasets.MNIST},
        {"name": "FashionMNIST", "batch_size": args.batch_size, "root_dir": conf.datasets.FashionMNIST},
    ]
    # Number of prototypes per class and number of trials for each number of prototype
    prototypes_trials = {n_proto: args.ntrials for n_proto in [1, 5, 10]}

    config = {
        "model_names": model_names,
        "datasets": datasets,
        "prototypes_trials": prototypes_trials,
        "override_models": []
    }

    run(main, config, load_results_id, accuracies={}, confusion_matrices={})
