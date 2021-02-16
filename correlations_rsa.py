import os
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.spatial.distance import squareform
from transformers import pipeline

from clip import tokenize
from utils import get_prototypes, get_model, get_dataset, language_model_features, save_corr_results, load_corr_results, \
    get_rdm, project_rdms

if __name__ == '__main__':
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    load_results_id = None

    # Models to test
    model_names = [
        "CLIP-RN50",
        "RN50",
        "virtex",
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
        {"name": "ImageNet", "batch_size": 64, "root_dir": os.path.expanduser("~/imagenet")},
        # {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "CIFAR10", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "CIFAR100", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        # {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
    ]
    # Number of prototypes per class and number of trials for each number of prototype
    prototypes_trials = {n_proto: 1 for n_proto in [-1]}

    caption_sentence_prototypes = [("a photo of {classname}.", 3), ("{classname}.", 0)]

    transformer_model = pipeline("feature-extraction", "bert-base-uncased")

    # features = np.array(transformer_model(["a photo of a dog"])).mean(axis=1)

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

    resnet_features = dict()

    if load_results_id is not None:
        rsa_corr_bert, rsa_corr_resnet, resnet_bert_score, loaded_config = load_corr_results(
            Path(f"results/{load_results_id}"))
    else:
        rsa_corr_bert = dict()
        rsa_corr_resnet = dict()
        resnet_bert_score = dict()

    try:
        with torch.no_grad():
            resnet_model, resnet_transform = get_model("RN50", device)

            for model_name in model_names:
                if model_name not in rsa_corr_bert:
                    rsa_corr_bert[model_name] = dict()
                if model_name not in rsa_corr_resnet:
                    rsa_corr_resnet[model_name] = dict()
                if model_name not in resnet_bert_score:
                    resnet_bert_score[model_name] = dict()
                # Import model
                model, transform = get_model(model_name, device)

                for dataset in datasets:
                    if dataset['name'] not in rsa_corr_bert[model_name]:
                        rsa_corr_bert[model_name][dataset['name']] = dict()
                    if dataset['name'] not in rsa_corr_resnet[model_name]:
                        rsa_corr_resnet[model_name][dataset['name']] = dict()
                    if dataset['name'] not in resnet_bert_score[model_name]:
                        resnet_bert_score[model_name][dataset['name']] = dict()

                    # Get dataset
                    _, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform)
                    _, dataset_resnet, _, _ = get_dataset(dataset, resnet_transform)

                    for n_proto in prototypes_trials.keys():
                        # Average results over several trials
                        bert_corr = []
                        resnet_corr = []
                        model_scores = []
                        bert_text_corr = None
                        resnet_text_corr = None
                        model_text_scores = None
                        n_trials = prototypes_trials[n_proto]

                        if model.has_text_encoder:
                            bert_text_corr = []
                            resnet_text_corr = []
                            model_text_scores = []
                            if model_name + '-text' not in rsa_corr_bert:
                                rsa_corr_bert[f'{model_name}-text'] = dict()
                                rsa_corr_resnet[f'{model_name}-text'] = dict()
                                resnet_bert_score[f'{model_name}-text'] = dict()
                            if dataset['name'] not in rsa_corr_bert[f'{model_name}-text']:
                                rsa_corr_bert[f'{model_name}-text'][dataset['name']] = dict()
                                rsa_corr_resnet[f'{model_name}-text'][dataset['name']] = dict()
                                resnet_bert_score[f'{model_name}-text'][dataset['name']] = dict()

                        if (model_name in rsa_corr_bert and
                                dataset['name'] in rsa_corr_bert[model_name] and
                                n_proto in rsa_corr_bert[model_name][dataset['name']] and
                                model_name in rsa_corr_resnet and
                                dataset['name'] in rsa_corr_resnet[model_name] and
                                n_proto in rsa_corr_resnet[model_name][dataset['name']]):
                            print(
                                f"{model_name}. {dataset['name']}. {n_proto} prototype(s). SKIPPED.")
                            continue

                        for k in range(n_trials):
                            print(
                                f"{model_name}. {dataset['name']}. {n_proto} prototype(s). Trial {k + 1} / {n_trials}.")

                            # Define class prototypes
                            class_features, class_features_std, class_feature_counts = get_prototypes(
                                model, dataset_test, device, n_examples_per_class=n_proto,
                                n_classes=len(class_names), batch_size=dataset['batch_size'])
                            if dataset['name'] not in resnet_features:
                                features, features_std, feature_counts = get_prototypes(
                                    resnet_model, dataset_resnet, device, n_examples_per_class=n_proto,
                                    n_classes=len(class_names), batch_size=dataset['batch_size'])
                                resnet_features[dataset['name']] = squareform(get_rdm(features, features_std, feature_counts),
                                                                              checks=False)

                            rdm_resnet = resnet_features[dataset['name']]
                            rdm_model = squareform(get_rdm(class_features, class_features_std, class_feature_counts), checks=False)
                            corr_image = scipy.stats.pearsonr(rdm_model, rdm_resnet)[0]
                            resnet_corr.append(corr_image)

                            for caption_prototype, class_token_position in caption_sentence_prototypes:
                                captions = [caption_prototype.format(classname=classname) for classname in class_names]
                                language_features = language_model_features(transformer_model,
                                                                            captions, class_token_position + caption_class_location).to(class_features)
                                rdm_language = squareform(get_rdm(language_features), checks=False)
                                corr_language = scipy.stats.pearsonr(rdm_model, rdm_language)[0]
                                bert_corr.append(corr_language)

                                # ResNet - Bert Score
                                model_scores.append(project_rdms(rdm_resnet, rdm_language, rdm_model))

                                if model.has_text_encoder:
                                    tokenized_captions = tokenize(captions).to(device)
                                    model_language_features = model.encode_text(tokenized_captions)
                                    rdm_model_language = squareform(get_rdm(model_language_features), checks=False)
                                    corr_model_language = scipy.stats.pearsonr(rdm_model_language, rdm_language)[0]
                                    bert_text_corr.append(corr_model_language)
                                    corr_model_language_im = scipy.stats.pearsonr(rdm_model_language, rdm_resnet)[0]
                                    resnet_text_corr.append(corr_model_language)

                                    model_text_scores.append(project_rdms(rdm_resnet, rdm_language, rdm_model_language))

                        rsa_corr_bert[model_name][dataset['name']][n_proto] = (np.mean(bert_corr, axis=0),
                                                                               np.std(bert_corr, axis=0))
                        rsa_corr_resnet[model_name][dataset['name']][n_proto] = (np.mean(resnet_corr, axis=0),
                                                                                 np.std(resnet_corr, axis=0))
                        resnet_bert_score[model_name][dataset['name']][n_proto] = (np.mean(model_scores, axis=0),
                                                                                   np.std(model_scores, axis=0))
                        if model.has_text_encoder:
                            rsa_corr_bert[f"{model_name}-text"][dataset['name']][n_proto] = (
                                np.mean(bert_text_corr, axis=0),
                                np.std(bert_text_corr, axis=0)
                            )
                            rsa_corr_resnet[f"{model_name}-text"][dataset['name']][n_proto] = (
                                np.mean(resnet_text_corr, axis=0),
                                np.std(resnet_text_corr, axis=0)
                            )
                            resnet_bert_score[f"{model_name}-text"][dataset['name']][n_proto] = (
                                np.mean(model_text_scores, axis=0),
                                np.std(model_text_scores, axis=0)
                            )

                        print("ResNet-Bert Score")
                        print(resnet_bert_score)
                        print("BERT comp")
                        print(rsa_corr_bert)
                        print("ResNet comp")
                        print(rsa_corr_resnet)

        print(rsa_corr_bert)
        print(rsa_corr_resnet)
        print(resnet_bert_score)
        save_corr_results(results_path, rsa_corr_bert, rsa_corr_resnet, resnet_bert_score, config)
    except Exception as e:
        print("An error occurred... Saving results so far.")
        save_corr_results(results_path, rsa_corr_bert, rsa_corr_resnet, resnet_bert_score, config)
        raise e
