import json
import os
from pathlib import Path

import numpy as np
import scipy
import torch
import umap
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE

from utils import get_prototypes, get_model, get_dataset, get_rdm


def load_corr_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    corr = np.load(results_path / "correlations.npy", allow_pickle=True).item()
    significance = dict()
    if (results_path / "correlations.npy").exists():
        significance = np.load(results_path / "correlations.npy", allow_pickle=True).item()
    feature_cache = np.load(results_path / "feature_cache.npy", allow_pickle=True).item()
    dim_reducted_features = {}
    if (results_path / "dim_reducted_features.npy").exists():
        dim_reducted_features = np.load(results_path / "dim_reducted_features.npy", allow_pickle=True).item()
    return corr, significance, feature_cache, dim_reducted_features, config


def save_corr_results(results_path, corr, significance, feature_cache, dim_reducted_features, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir(exist_ok=True)
    np.save(str(results_path / "correlations.npy"), corr)
    np.save(str(results_path / "significance.npy"), significance)
    np.save(str(results_path / "feature_cache.npy"), feature_cache)
    np.save(str(results_path / "dim_reducted_features.npy"), dim_reducted_features)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_results_id = 173

    # Models to test
    model_names = [
        # "semi-supervised-YFCC100M",
        # "semi-weakly-supervised-instagram",
        "madry-imagenet_l2_3_0",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
        "geirhos-resnet50_trained_on_SIN",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "BERT",
        "GPT2",
        # "Word2Vec",
        "CLIP-RN50",
        "RN50",
        "virtex",
        "BiT-M-R50x1",
    ]
    # Dataset to test on
    datasets = [
        {"name": "ImageNet", "batch_size": 64, "root_dir": "/mnt/SSD/datasets/imagenet"},
        # {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "CIFAR10", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "CIFAR100", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        # {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
    ]

    caption_sentence_prototypes = ("a photo of {classname}.", 3)

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
        correlations, significance, model_features_cache, dim_reducted_features, loaded_config = load_corr_results(Path(f"results/{load_results_id}"))
    else:
        correlations = dict()
        significance = dict()
        model_features_cache = {}
        dim_reducted_features = {
            "umap": None,
            "tsne": None,
            "labels": None
        }

    items_to_remove = [
        # "geirhos-resnet50_trained_on_SIN",
        # "geirhos-resnet50_trained_on_SIN_and_IN",
        # "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        # "madry-imagenet_l2_3_0",
        # "madry-imagenet_linf_4",
        # "madry-imagenet_linf_8",
        # "virtex",
    ]
    for item in items_to_remove:
        del model_features_cache[item]

    try:
        with torch.no_grad():
            for model_name in model_names:
                # Import model
                model, transform = get_model(model_name, device)

                for dataset in datasets:
                    # Get dataset
                    _, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform)

                    print(f"{model_name}. {dataset['name']}.")

                    if model.has_image_encoder and model_name not in model_features_cache:
                        # Define class prototypes
                        class_features, class_features_std, class_feature_counts = get_prototypes(
                            model, dataset_test, device, n_examples_per_class=-1,
                            n_classes=len(class_names), batch_size=dataset['batch_size']
                        )
                        rdm_model = squareform(get_rdm(class_features, class_features_std, class_feature_counts),
                                               checks=False)
                        model_features_cache[model_name] = rdm_model

                    if model.has_text_encoder and model_name + "_text" not in model_features_cache:
                        caption_prototype, class_token_position = caption_sentence_prototypes
                        captions = [caption_prototype.format(classname=classname) for classname in class_names]
                        model_language_features = model.encode_text(captions, device, class_token_position + caption_class_location)
                        rdm_model_language = squareform(get_rdm(model_language_features), checks=False)
                        model_features_cache[model_name + "_text"] = rdm_model_language

            # Compute correlations between all models
            for model_1, rdm_model_1 in model_features_cache.items():
                correlations[model_1] = {}
                significance[model_1] = {}
                for model_2, rdm_model_2 in model_features_cache.items():
                    r, p = scipy.stats.pearsonr(rdm_model_1, rdm_model_2)
                    correlations[model_1][model_2] = r
                    significance[model_1][model_2] = p

        y, X = zip(*model_features_cache.items())
        X = np.stack(X, axis=0)
        reducer = umap.UMAP(n_components=2, min_dist=0.05, n_neighbors=5, metric="correlation")
        x_umap = reducer.fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=3, learning_rate=1, min_grad_norm=0, metric="correlation")
        X_tsne = tsne.fit_transform(X)

        dim_reducted_features = {
            "umap": x_umap,
            "tsne": X_tsne,
            "labels": y
        }

        print(correlations)
        print(significance)
        save_corr_results(results_path, correlations, significance, model_features_cache, dim_reducted_features, config)
    except Exception as e:
        print("An error occurred... Saving results so far.")
        save_corr_results(results_path, correlations, significance, model_features_cache, dim_reducted_features, config)
        raise e
