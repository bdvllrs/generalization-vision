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
    """
    Returns all features and labels of a dataset for a given model.
    Args:
        model_: model to compute over. Should have an image encoder.
        dataset:
        device:
        batch_size:

    Returns: Couple (features, labels) where
        features: (N_IMAGES, DIM_FEAT) where N_IMAGES is the number of images in the dataset
        labels: (N_IMAGES,) the labels for each feature.

    """
    assert model_.has_image_encoder, "Model should have an image encoder."
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)
    dataloader_iter = iter(dataloader)

    features = []
    labels = []

    for image_input, target in tqdm(dataloader_iter, total=len(dataloader_iter)):
        # Get feature from image
        feature = model_.encode_image(image_input.to(device))
        # normalize by the norm of the vector to compute the dot product later on.
        feature /= feature.norm(dim=-1, keepdim=True)
        features.append(feature.detach().cpu().numpy())
        labels.extend(target.tolist())
    return np.concatenate(features, axis=0), np.array(labels)


def get_prototypes(model_, train_set, device, n_examples_per_class=5, n_classes=10, *params, **kwargs):
    """
    Returns class prototypes of a given model and dataset.
    Args:
        model_: model to use. Should have the encode_image method.
        train_set:
        device:
        n_examples_per_class: number of examples to use to compute the prototypes. -1 for all images of the dataset.
        n_classes: Number of classes in the dataset.
        *params: Extra parameters of get_set_features in the case of -1 for n_examples_per_class.
        **kwargs: Extra parameters of get_set_features in the case of -1 for n_examples_per_class.

    Returns:

    """
    assert model_.has_image_encoder, "Model should have an image encoder."
    assert n_examples_per_class >= 1 or n_examples_per_class == -1

    # If we use all the images
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

    # If we use a subset of image, start by selecting a few
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
    # The compute mean and std
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
    """
    Unequal variance t-test (see https://en.wikipedia.org/wiki/Welch%27s_t-test)
    """
    return (x - y) / np.sqrt(np.square(x_std) / count_x + np.square(y_std) / count_y)


def get_rdm(features, feature_std=None, feature_counts=None):
    """
    Computed the rdm given a features matrix
    Args:
        features: Matrix of features (N_classes, Feat_dim). Features are the mean over all image/text examples.
        feature_std: std of the class examples (std associated to the means given by features)
        feature_counts: count of examples per class (e.g. 2 if 2 images are used to compute the features of the class)

    Returns: RDM matrix
    """
    # Use numpy arrays
    features = features.cpu().numpy()
    if feature_std is not None:
        feature_std = feature_std.cpu().numpy()
        feature_counts = feature_counts.cpu().numpy()
    rdm = np.zeros((features.shape[0], features.shape[0]))
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            if feature_std is not None:
                # then use the t-test distance
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


def load_results(results_path: Path):
    """
    Load checkpoint
    Args:
        results_path: path to checkpoint

    Returns: loaded config and dictionary of all .npy file saved for this experiment.
    """
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    params = {}
    for file in results_path.iterdir():
        if file.suffix == ".npy":
            params[file.stem] = np.load(str(file), allow_pickle=True).item()
    return config, params


def save_results(results_path, config, **params):
    """
    Save checkpoints to npy file.
    Args:
        results_path: path to checkpoint
        config: configuration of experiment
        **params: files to save. One .npy file will be crated by element in the params dict.
    """
    results_path = Path(results_path)
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir(exist_ok=True)
    for name, val in params.items():
        np.save(str(results_path / f"{name}.npy"), val)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


def run(fun, config: dict, load_saved_results: int = None, **params):
    """
    Run the task
    Args:
        fun: main function to execute. Takes parameters: config, **params
        config: config dict. Saved in checkpoints.
        load_saved_results: An id of previous experiment to load the config and **params from.
            the *provided* config overrides the loaded one; the *loaded* **params overrides the provided ones.
        **params: additional parameters to save.
    """

    # Make directories to save checkpoints.
    results_path = Path("../results")
    results_path.mkdir(exist_ok=True)

    # compute new id of the experiment
    existing_folders = [int(f.name) for f in results_path.glob("*") if f.is_dir() and f.name.isdigit()]
    result_idx = max(existing_folders) + 1 if len(existing_folders) else 0

    results_path = results_path / str(result_idx)
    results_path.mkdir()

    # load the config and params if id is provided.
    loaded_config = config
    if load_saved_results is not None:
        loaded_config, loaded_results = load_results(Path(f"results/{load_saved_results}"))
        params.update(loaded_results)
        loaded_config.update(config)

    loaded_config["results_path"] = str(results_path)
    # run main
    fun(loaded_config, **params)
