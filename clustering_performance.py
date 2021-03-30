import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from utils import get_model, get_dataset, get_set_features


def load_corr_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    acc = np.load(results_path / "acc.npy", allow_pickle=True).item()
    return acc, config


def save_corr_results(results_path, acc, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir(exist_ok=True)
    np.save(str(results_path / "acc.npy"), acc)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


def permute_labels(target, prediction, num_labels):
    congruence = np.zeros((num_labels, num_labels))
    for label in range(num_labels):
        for perm in range(num_labels):
            congruence[label, perm] = np.logical_and(target == label, prediction == perm).sum()
    perm = np.zeros(num_labels, dtype=int)
    done_labels = []
    ind_max = np.unravel_index(np.argsort(-congruence, axis=None), congruence.shape)
    for label1, label2 in zip(*ind_max):
        if len(done_labels) == num_labels:
            break
        if label1 not in done_labels:
            perm[label1] = label2
            done_labels.append(label1)
    permuted_labels = np.zeros_like(prediction)
    for label in range(num_labels):
        permuted_labels[prediction == perm[label]] = label
    return permuted_labels


if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    load_results_id = 46

    # Models to test
    model_names = [
        "semi-supervised-YFCC100M",
        "semi-weakly-supervised-instagram",
        "madry-imagenet_l2_3_0",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
        "geirhos-resnet50_trained_on_SIN",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "virtex",
        "CLIP-RN50",
        "RN50",
        "BiT-M-R50x1",
    ]
    # Dataset to test on
    datasets = [
        # {"name": "ImageNet", "batch_size": 64, "root_dir": os.path.expanduser("~/imagenet")},
        {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR10", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR100", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
    ]

    # Make directories to save data
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)

    existing_folders = [int(f.name) for f in results_path.glob("*") if f.is_dir() and f.name.isdigit()]
    result_idx = max(existing_folders) + 1 if len(existing_folders) else 0

    results_path = results_path / str(result_idx)
    results_path.mkdir()

    config = {
        "model_names": model_names,
        "datasets": datasets,
    }

    if load_results_id is not None:
        acc, loaded_config = load_corr_results(Path(f"results/{load_results_id}"))
    else:
        acc = dict()

    items_to_remove = [
        "geirhos-resnet50_trained_on_SIN",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "madry-imagenet_l2_3_0",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
    ]
    for item in items_to_remove:
        del acc[item]

    try:
        with torch.no_grad():
            for model_name in model_names:
                # Import model
                model, transform = get_model(model_name, device)

                if model_name not in acc:
                    acc[model_name] = {}

                for dataset in datasets:
                    # Get dataset
                    dataset_train, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform)

                    print(f"{model_name}. {dataset['name']}.")

                    assert model.has_image_encoder, f"{model_name} has no image encoder."

                    if dataset['name'] not in acc[model_name]:
                        # if len(dataset_train) > 40_000:
                        #     indices = np.arange(len(dataset_train))
                        #     np.random.shuffle(indices)
                        #     dataset_train = torch.utils.data.Subset(dataset_train, indices[:40_000])
                        # dataset_all = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
                        # Define class prototypes
                        all_features, labels = get_set_features(model, dataset_test, device,
                                                                batch_size=dataset['batch_size'])
                        clustering_algorithm = AgglomerativeClustering(n_clusters=len(class_names),
                                                                  affinity="cosine",
                                                                  linkage="average")
                        predicted_labels = clustering_algorithm.fit_predict(all_features)
                        # Only keep val labels
                        predicted_labels = predicted_labels[len(dataset_train):]
                        labels = labels[len(dataset_train):]
                        # Compute accuracy
                        permuted_predicted_labels = permute_labels(labels, predicted_labels, len(class_names))
                        acc[model_name][dataset['name']] = (permuted_predicted_labels == labels).sum() / labels.shape[0]

        print(acc)
        save_corr_results(results_path, acc, config)
    except Exception as e:
        print("An error occurred... Saving results so far.")
        save_corr_results(results_path, acc, config)
        raise e
