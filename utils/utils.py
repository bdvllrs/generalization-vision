import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as confusion_matrix_, accuracy_score as accuracy_score_
from torch.utils import model_zoo
from tqdm import tqdm



def plot_class_predictions(images, class_names, probs,
                           square_size=4, show_best_n_classes=5):
    """
    Plot the images and topk predictions
    """
    n_columns = 2 if len(images) == 1 else 4
    plt.figure(figsize=(n_columns * square_size, square_size * len(images) // 2))

    images -= images.min()
    images /= images.max()

    n_top = min(show_best_n_classes, len(class_names))
    top_probs, top_labels = probs.float().cpu().topk(n_top, dim=-1)

    for i, image in enumerate(images):
        plt.subplot(len(images) // 2 + 1, n_columns, 2 * i + 1)
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(len(images) // 2 + 1, n_columns, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [class_names[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()


def averaged_text_features(model_, texts):
    # Calculate features
    with torch.no_grad():
        # Combine text representations
        text_features = torch.stack([model_.encode_text(text_inputs) for text_inputs in texts], dim=0)
        text_features = text_features.mean(dim=0)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def evaluate_dataset(model_, dataset, text_inputs, labels, device, batch_size=64, encode_text=True):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    dataloader_iter = iter(dataloader)
    predictions = []
    targets = []

    with torch.no_grad():
        if encode_text:
            if type(text_inputs) == list:
                # average of several prompts
                text_features = averaged_text_features(model_, text_inputs)
            else:
                text_features = model_.encode_text(text_inputs.to(device))
                text_features /= text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = text_inputs
        for image_input, target in tqdm(dataloader_iter, total=len(dataloader_iter)):
            image_features = model_.encode_image(image_input.to(device))

            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T

            predicted_class = similarity.max(dim=-1).indices
            targets.append(target)
            predictions.append(predicted_class)

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
    return (accuracy_score_(targets.cpu(), predictions.cpu()),
            confusion_matrix_(targets.cpu(), predictions.cpu(), labels=labels))


def get_set_features(model_, dataset, device, batch_size=64):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    dataloader_iter = iter(dataloader)

    features = []
    labels = []

    for image_input, target in tqdm(dataloader_iter, total=len(dataloader_iter)):
        feature = model_.encode_image(image_input.to(device))
        feature /= feature.norm(dim=-1, keepdim=True)
        features.append(feature.detach().cpu().numpy())
        labels.extend(target.tolist())
    return np.concatenate(features, axis=0), np.array(labels)


def get_prototypes(model_, train_set, device, n_examples_per_class=5, n_classes=10, *params, **kwargs):
    assert n_examples_per_class >= 1 or n_examples_per_class == -1

    if n_examples_per_class == -1:
        all_features, labels = get_set_features(model_, train_set, device, *params, **kwargs)
        features = np.zeros((n_classes, all_features.shape[1]))
        std = np.zeros((n_classes, 1))
        counts = np.zeros((n_classes, 1))
        for k in range(n_classes):
            features[k] = np.mean(all_features[labels == k], axis=0)
            std[k] = np.std(all_features[labels == k])
            counts[k] = np.sum(labels == k)
        return torch.from_numpy(features), torch.from_numpy(std), torch.from_numpy(counts)

    prototypes = [[] for k in range(n_classes)]
    done = []
    for image, label in iter(train_set):
        if len(prototypes[label]) < n_examples_per_class:
            prototypes[label].append(image)
        if label not in done and len(prototypes[label]) == n_examples_per_class:
            done.append(label)
        if len(done) == n_classes:
            break
    features = []
    std = []
    for proto_imgs in prototypes:
        imgs = torch.stack(proto_imgs, dim=0).to(device)
        feature = model_.encode_image(imgs)
        feature /= feature.norm(dim=-1, keepdim=True)
        feat_mean = feature.mean(0)
        feat_var = feature.std(0)
        features.append(feat_mean)
        std.append(feat_var)
    return torch.stack(features, dim=0), torch.stack(std, dim=0), torch.ones(len(std)).fill_(n_examples_per_class)

def t_test(x, y, x_std, y_std, count_x, count_y):
    return (x - y) / np.sqrt(np.square(x_std) / count_x + np.square(y_std) / count_y)

def get_rdm(features, feature_std=None, feature_counts=None):
    features = features.cpu().numpy()
    if feature_std is not None:
        feature_std = feature_std.cpu().numpy()
        feature_counts = feature_counts.cpu().numpy()
    rdm = np.zeros((features.shape[0], features.shape[0]))
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            if feature_std is not None:
                rdm[i, j] = np.linalg.norm(t_test(features[i], features[j], feature_std[i],
                                                  feature_std[j], feature_counts[i], feature_counts[j]))
            else:
                rdm[i, j] = np.linalg.norm(features[i] - features[j])
            rdm[j, i] = rdm[i, j]
    return rdm


def language_model_features(language_model, tokenizer, captions, class_token_position=0):
    inputs = tokenizer(captions)
    hidden_states = language_model(inputs, output_hidden_states=True)['hidden_states']
    return hidden_states[class_token_position + 1]  # +1 for start token


def cca_plot_helper(arr, xlabel, ylabel):
    plt.plot(arr, lw=2.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()


def clip_encode_text_no_projection(clip_, tokens):
    x = clip_.token_embedding(tokens).type(clip_.dtype)  # [batch_size, n_ctx, d_model]

    x = x + clip_.positional_embedding.type(clip_.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_.ln_final(x).type(clip_.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
    return x


def project_rdms(rdm_resnet, rdm_bert, rdm_model):
    rdm_bert_recentered = (rdm_bert - rdm_resnet).reshape(-1, 1)
    norm = (rdm_bert_recentered.T @ rdm_bert_recentered)
    # Project the vector onto the [resnet, BERT] axis
    projected = rdm_bert_recentered @ (rdm_bert_recentered.T @ (rdm_model - rdm_resnet).reshape(-1, 1)) / norm
    # Compare norms of the vectors
    rdm_bert_recentered_norm = np.linalg.norm(rdm_bert_recentered)
    score = np.dot(projected[:, 0], rdm_bert_recentered[:, 0]) / (rdm_bert_recentered_norm * rdm_bert_recentered_norm)
    return score


def old_save_results(results_path, accuracies, confusion_matrices, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir(exist_ok=True)
    np.save(str(results_path / "accuracies.npy"), accuracies)
    np.save(str(results_path / "confusion_matrices.npy"), confusion_matrices)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


def save_corr_results(results_path, bert_corr, resnet_corr, resnet_bert_score, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir()
    np.save(str(results_path / "bert_corr.npy"), bert_corr)
    np.save(str(results_path / "resnet_corr.npy"), resnet_corr)
    np.save(str(results_path / "resnet_bert_score.npy"), resnet_bert_score)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


def old_load_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    accuracies = np.load(results_path / "accuracies.npy", allow_pickle=True).item()
    confusion_matrices = np.load(results_path / "confusion_matrices.npy", allow_pickle=True).item()
    return accuracies, confusion_matrices, config

def load_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    for file in results_path.iter():
        if file.name().split(".") == "npy":
            print('ok')
    # accuracies = np.load(results_path / "accuracies.npy", allow_pickle=True).item()
    # confusion_matrices = np.load(results_path / "confusion_matrices.npy", allow_pickle=True).item()
    # return accuracies, confusion_matrices, config
    return config, {}

def save_results(results_path, config, **params):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir(exist_ok=True)
    for name, val in params.items():
        np.save(str(results_path / f"{name}.npy"), val)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

def load_corr_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    bert_corr = np.load(results_path / "bert_corr.npy", allow_pickle=True).item()
    resnet_corr = np.load(results_path / "resnet_corr.npy", allow_pickle=True).item()
    resnet_bert_score = np.load(results_path / "resnet_bert_score.npy", allow_pickle=True).item()
    return bert_corr, resnet_corr, resnet_bert_score, config


def run(fun, config, results_path, load_saved_results=False, **params):
    if load_saved_results:
        config, loaded_results = load_results(results_path)
        params.update(loaded_results)
    try:
        fun()
        save_results(results_path, config, **params)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(results_path, config, **params)
        raise e
