import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import pipeline

from svcca import cca_core
from utils import get_set_features, language_model_features, cca_plot_helper
from utils.datasets.datasets import get_dataset
from utils.models import get_model

if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    load_results_id = 7

    # Models to test
    model_names = [
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
        {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR10", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR100", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
    ]
    # Number of prototypes per class and number of trials for each number of prototype
    prototypes_trials = {n_proto: 10 for n_proto in [1, 5, 10]}
    num_singular_values = 5

    # list of (caption, position of class token)
    caption_sentence_prototypes = [("a photo of {classname}.", 3), ("{classname}.", 0)]

    transformer_model = pipeline("feature-extraction", "bert-base-uncased")

    features = np.array(transformer_model(["a photo of a dog"])).mean(axis=1)

    corr_to_bert = {}
    corr_to_resnet = {}

    with torch.no_grad():
        resnet_model, resnet_transform = get_model("RN50", device)

        for model_name in model_names:
            if model_name not in corr_to_bert:
                corr_to_bert[model_name] = dict()
            if model_name not in corr_to_resnet:
                corr_to_resnet[model_name] = dict()
            # Import model
            model, transform = get_model(model_name, device)

            for dataset in datasets:
                if dataset['name'] not in corr_to_bert[model_name]:
                    corr_to_bert[model_name][dataset['name']] = dict()
                if dataset['name'] not in corr_to_resnet[model_name]:
                    corr_to_resnet[model_name][dataset['name']] = dict()
                # Get dataset
                _, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform)
                _, dataset_test_resnet, _, _ = get_dataset(dataset, resnet_transform)

                for n_proto in prototypes_trials.keys():
                    # Average results over several trials
                    trial_corr_to_bert = []
                    trial_corr_to_resnet = []
                    n_trials = prototypes_trials[n_proto]

                    for k in range(n_trials):
                        print(
                            f"{model_name}. {dataset['name']}. {n_proto} prototype(s). Trial {k + 1} / {n_trials}.")

                        # Define class prototypes
                        dataset_features, labels = get_set_features(model, dataset_test, device,
                                                                    batch_size=dataset['batch_size'])
                        resnet_features, _ = get_set_features(resnet_model, dataset_test_resnet, device,
                                                              batch_size=dataset['batch_size'])
                        results = cca_core.get_cca_similarity(dataset_features.transpose(),
                                                              resnet_features.transpose(),
                                                              threshold=0.98,
                                                              epsilon=1e-6, verbose=False)
                        cca_plot_helper(results['cca_coef1'], "CCA coef idx", "CCA coef value")
                        plt.show()
                        corr = np.mean(results['cca_coef1'][:num_singular_values])
                        trial_corr_to_resnet.append(corr)

                        for caption_prototype, class_token_position in caption_sentence_prototypes:
                            captions = [caption_prototype.format(classname=classname) for classname in class_names]
                            captions = language_model_features(transformer_model, captions, class_token_position + caption_class_location)
                            dataset_language_features = np.zeros((dataset_features.shape[0], captions.shape[1]))
                            for j in range(len(class_names)):
                                dataset_language_features[labels == j] = captions[j]
                            # dataset_language_features += 1e-4 * np.random.randn(*dataset_language_features.shape)

                            results = cca_core.get_cca_similarity(dataset_features.transpose(),
                                                                  dataset_language_features.transpose(),
                                                                  threshold=0.98,
                                                                  epsilon=1e-6, verbose=False)
                            # cca_plot_helper(results['cca_coef1'], "CCA coef idx", "CCA coef value")
                            # plt.show()
                            corr = np.mean(results['cca_coef1'][:num_singular_values])
                            trial_corr_to_bert.append(corr)

                    corr_to_bert[model_name][dataset['name']][n_proto] = (np.mean(trial_corr_to_bert, axis=0),
                                                                        np.std(trial_corr_to_bert, axis=0))
                    corr_to_resnet[model_name][dataset['name']][n_proto] = (np.mean(trial_corr_to_resnet, axis=0),
                                                                        np.std(trial_corr_to_resnet, axis=0))

