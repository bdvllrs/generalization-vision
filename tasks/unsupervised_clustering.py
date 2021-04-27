import os

import argparse
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from visiongeneralization.datasets.datasets import get_dataset
from visiongeneralization.models import get_model
from visiongeneralization.utils import get_set_features, save_results, run


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


def main(config, acc):
    try:
        model_names, datasets = config["model_names"], config["datasets"]
        override_models = config["override_models"]

        for item in override_models:
            del acc[item]

        with torch.no_grad():
            for model_name in model_names:
                # Import model
                model, transform = get_model(model_name, device)
                model.eval()

                # initialize results dict
                if model_name not in acc:
                    acc[model_name] = {}

                for dataset in datasets:
                    # Get dataset
                    dataset_train, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform)

                    print(f"{model_name}. {dataset['name']}.")

                    assert model.has_image_encoder, f"{model_name} has no image encoder."

                    # check that data has not yet been computed
                    if dataset['name'] not in acc[model_name]:
                        # Define class prototypes
                        all_features, labels = get_set_features(model, dataset_test, device,
                                                                batch_size=dataset['batch_size'])
                        clustering_algorithm = AgglomerativeClustering(n_clusters=len(class_names),
                                                                       affinity="cosine",
                                                                       linkage="average")
                        predicted_labels = clustering_algorithm.fit_predict(all_features)
                        # Label clusters to maximize accuracy
                        permuted_predicted_labels = permute_labels(labels, predicted_labels, len(class_names))
                        # compute accuracy
                        acc[model_name][dataset['name']] = (permuted_predicted_labels == labels).sum() / labels.shape[0]

        print(acc)
        save_results(config["results_path"], config, acc=acc)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(config["results_path"], config, acc=acc)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised clustering task.')
    parser.add_argument('--load_results', default=None, type=int,
                        help='Id of a previous experiment to continue.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size.')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models to test
    model_names = [
        "RN50",
        "CLIP-RN50",
        "BiT-M-R50x1",
        "virtex",
        "madry-imagenet_l2_3_0",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "semi-supervised-YFCC100M",
        "semi-weakly-supervised-instagram",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
        "geirhos-resnet50_trained_on_SIN",
        "geirhos-resnet50_trained_on_SIN_and_IN",
    ]
    # Dataset to test on
    datasets = [
        # {"name": "ImageNet", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/imagenet")},
        {"name": "MNIST", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR10", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR100", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "FashionMNIST", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "HouseNumbers", "batch_size": args.batch_size, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        {"name": "CUB", "batch_size": args.batch_size, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
    ]

    config = {
        "model_names": model_names,
        "datasets": datasets,
        "override_models": []
    }

    run(main, config, args.load_results, acc={})