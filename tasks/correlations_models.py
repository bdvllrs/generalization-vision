import json
from pathlib import Path

import argparse
import numpy as np
import scipy
import torch
import umap
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE

from visiongeneralization.utils import get_prototypes, get_rdm, save_results, run
from visiongeneralization.datasets.datasets import get_dataset
from visiongeneralization.models import get_model



def main(config, feature_cache, correlations, significance=None, dim_reducted_features=None):
    try:
        model_names, datasets = config["model_names"], config["datasets"]
        override_models = config["override_models"]
        caption_sentence_prototypes = config["caption_sentence_prototypes"]

        for item in override_models:
            del feature_cache[item]

        with torch.no_grad():
            for model_name in model_names:
                # Import model
                model, transform = get_model(model_name, device)
                model.eval()

                for dataset in datasets:
                    # Get dataset
                    _, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform)

                    print(f"{model_name}. {dataset['name']}.")

                    # For vision models
                    if model.has_image_encoder and model_name not in feature_cache:
                        # Define class prototypes
                        class_features, class_features_std, class_feature_counts = get_prototypes(
                            model, dataset_test, device, n_examples_per_class=-1,
                            n_classes=len(class_names), batch_size=dataset['batch_size']
                        )
                        rdm_model = squareform(get_rdm(class_features, class_features_std, class_feature_counts),
                                               checks=False)
                        feature_cache[model_name] = rdm_model

                    # For language models
                    if model.has_text_encoder and model_name + "_text" not in feature_cache:
                        caption_prototype, class_token_position = caption_sentence_prototypes
                        # Add the classnames to the captions
                        captions = [caption_prototype.format(classname=classname) for classname in class_names]
                        model_language_features = model.encode_text(captions, device,
                                                                    class_token_position + caption_class_location)
                        rdm_model_language = squareform(get_rdm(model_language_features), checks=False)
                        feature_cache[model_name + "_text"] = rdm_model_language

            # Compute correlations between all models
            for model_1, rdm_model_1 in feature_cache.items():
                correlations[model_1] = {}
                significance[model_1] = {}
                for model_2, rdm_model_2 in feature_cache.items():
                    r, p = scipy.stats.pearsonr(rdm_model_1, rdm_model_2)
                    correlations[model_1][model_2] = r
                    significance[model_1][model_2] = p

        # Compute UMAP and TSNE projections
        y, X = zip(*feature_cache.items())
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

        save_results(config["results_path"], config,
                     feature_cache=feature_cache, correlations=correlations, significance=significance,
                     dim_reducted_features=dim_reducted_features)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(config["results_path"], config,
                     feature_cache=feature_cache, correlations=correlations, significance=significance,
                     dim_reducted_features=dim_reducted_features)
        raise e


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Correlations between models.')
    parser.add_argument('--load_results', default=None, type=int,
                        help='Id of a previous experiment to continue.')
    parser.add_argument('--batch_size', default=80, type=int,
                        help='Batch size.')

    args = parser.parse_args()

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
        {"name": "ImageNet", "batch_size": args.batch_size, "root_dir": "/mnt/SSD/datasets/imagenet"},
    ]

    config = {
        "model_names": model_names,
        "datasets": datasets,
        "caption_sentence_prototypes": ("a photo of {classname}.", 3),
        "override_models": []
    }

    run(main, config, args.load_results,
        feature_cache={}, correlations={}, significance={}, dim_reducted_features={})
