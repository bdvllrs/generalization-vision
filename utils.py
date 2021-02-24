import json
import os
from pathlib import Path

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from robustness.imagenet_models import resnet50
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix as confusion_matrix_, accuracy_score as accuracy_score_
from torch.utils import model_zoo
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageNet
from torchvision.models import resnet50
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline

import clip
from bit_model import KNOWN_MODELS as BiT_MODELS

BiT_model_urls = {
    'BiT-M-R50x1': os.path.expanduser("~/.cache/torch/checkpoints/BiT-M-R50x1.npz"),
}

clip_models = ["ViT-B/32", "RN50"]

geirhos_model_urls = {
    'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
    'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
    # 'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
}

madry_model_folder = os.path.join(os.getenv("TORCH_HOME", "~/.cache/torch"), "checkpoints")
madry_models = ["imagenet_l2_3_0", "imagenet_linf_4", "imagenet_linf_8"]

imagenet_norm_mean, imagenet_norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
imagenet_transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_norm_mean, imagenet_norm_std)
])


class ModelEncapsulation(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        self.has_text_encoder = False
        self.has_image_encoder = True

    def encode_image(self, images):
        return self.module(images)


class CLIPLanguageModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
        self.has_text_encoder = True
        self.has_image_encoder = True

    def encode_image(self, image):
        return self.module.encode_image(image)

    def encode_text(self, text, device, class_token_position=0):
        text = clip.tokenize(text).to(device)
        return self.module.encode_text(text)


class GPT2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer_model = AutoModelWithLMHead.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, text, device, class_token_position=0):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        hidden_states = self.transformer_model(**inputs, output_hidden_states=True)['hidden_states']
        return hidden_states[class_token_position + 1]  # +1 for start token


class BERTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer_model = pipeline("feature-extraction", "bert-base-uncased")

        self.has_text_encoder = True
        self.has_image_encoder = False

    def encode_text(self, text, device, class_token_position=0):
        embedding = torch.tensor(self.transformer_model(text))
        return torch.tensor(embedding)[:, class_token_position + 1]  # +1 for start token


class RandomizedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

        self.order = np.arange(len(self.dataset))

    def randomize(self):
        self.order = np.random.permutation(len(self.dataset))
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[self.order[item]]


class CUBDataset:
    def __init__(self, root_dir, train=True, transform=None):
        split_idx = 1 if train else 0
        self.root_dir = Path(root_dir)
        self.id_file = self.root_dir / "images.txt"
        self.label_file = self.root_dir / "image_class_labels.txt"
        self.split_file = self.root_dir / "train_test_split.txt"
        self.transform = transform

        with open(self.split_file, "r") as f:
            self.split = [int(line.rstrip('\n').split(" ")[0]) for line in f if
                          int(line.rstrip('\n').split(" ")[1]) == split_idx]
        with open(self.label_file, "r") as f:
            self.labels = {int(line.rstrip('\n').split(" ")[0]): int(line.rstrip('\n').split(" ")[1]) for line in f if
                           int(line.rstrip('\n').split(" ")[0]) in self.split}
        with open(self.id_file, "r") as f:
            self.location = {int(line.rstrip('\n').split(" ")[0]): " ".join(line.rstrip('\n').split(" ")[1:]) for line
                             in f if int(line.rstrip('\n').split(" ")[0]) in self.split}

    def __len__(self):
        return len(self.split)

    def __getitem__(self, item):
        file_idx = self.split[item]
        label = self.labels[file_idx] - 1  # Start index at 0
        with Image.open(self.root_dir / "images" / self.location[file_idx]) as image:
            processed_image = self.transform(image)
        return processed_image, label


class HouseNumbersDataset:
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        image_path = "train_32x32.mat" if train else "test_32x32.mat"
        self.image_path = self.root_dir / image_path
        images = loadmat(str(self.image_path))
        self.images = images['X'].transpose((3, 0, 1, 2))
        self.labels = images['y']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item, 0]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        if label == 10:  # set index 0 for image of 0
            label = 0
        return image, label


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
        features.append(feature.mean(0))
        std.append(feature)
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


def get_model(model_name, device):
    if "CLIP" in model_name and model_name.replace("CLIP-", "") in clip_models:
        model, transform = clip.load(model_name.replace("CLIP-", ""), device=device, jit=False)
        model = CLIPLanguageModel(model)
    elif model_name == "RN50":
        resnet = resnet50(pretrained=True)
        resnet.fc = torch.nn.Identity()  # remove last linear layer before softmax function
        model = ModelEncapsulation(resnet)
        model = model.to(device)
        transform = imagenet_transform
    elif model_name == "virtex":
        model = ModelEncapsulation(torch.hub.load("kdexd/virtex", "resnet50", pretrained=True))
        model.to(device)
        transform = imagenet_transform
    elif "geirhos" in model_name and model_name.replace("geirhos-", "") in geirhos_model_urls.keys():
        model = resnet50(pretrained=False)
        checkpoint = model_zoo.load_url(geirhos_model_urls[model_name.replace("geirhos-", "")],
                                        map_location=torch.device('cpu'))
        model = ModelEncapsulation(model)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        transform = imagenet_transform
    elif "madry" in model_name and model_name.replace("madry-", "") in madry_models:
        model = resnet50(pretrained=False)
        checkpoint = torch.load(os.path.join(madry_model_folder, model_name.replace("madry-", "") + ".pt"),
                                map_location=torch.device('cpu'))['model']
        checkpoint = {mod_name.replace(".model", ""): mod_param for mod_name, mod_param in checkpoint.items() if
                      "module.model" in mod_name}
        model = ModelEncapsulation(model)
        model.load_state_dict(checkpoint)
        model.to(device)
        transform = imagenet_transform
    elif "BiT" in model_name and model_name in BiT_model_urls:
        model = BiT_MODELS[model_name]()
        model.load_from(np.load(BiT_model_urls[model_name]))
        model = ModelEncapsulation(model)
        model.to(device)
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # Language models
    elif model_name == "BERT":
        model = BERTModel()
        transform = None
    elif model_name == "GPT2":
        model = GPT2Model()
        transform = None
    elif model_name == "Word2Vec":
        #TODO
        raise ValueError(f"{model_name} is not a valid model name.")
    else:
        raise ValueError(f"{model_name} is not a valid model name.")
    return model, transform


def get_dataset(dataset, transform):
    caption_class_position = 1
    if dataset['name'] == "MNIST":
        # Download the dataset
        dataset_train = MNIST(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = MNIST(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        class_names = [f"the number {classname}" for classname in class_names]
        caption_class_position = 2
    elif dataset['name'] == "FashionMNIST":
        dataset_train = FashionMNIST(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = FashionMNIST(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = [f"a {class_name}" for class_name in map(lambda x: x.lower(), dataset_test.classes)]
    elif dataset['name'] == "CIFAR10":
        dataset_train = CIFAR10(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = CIFAR10(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = [f"a {class_name}" for class_name in map(lambda x: x.lower(), dataset_test.classes)]
    elif dataset['name'] == "CIFAR100":
        dataset_train = CIFAR100(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = CIFAR100(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = [f"a {class_name}" for class_name in map(lambda x: x.lower(), dataset_test.classes)]
    elif dataset['name'] == "ImageNet":
        dataset_train = ImageNet(root=dataset["root_dir"], split='train', transform=transform)
        dataset_test = ImageNet(root=dataset["root_dir"], split='val', transform=transform)
        class_names = [f"a {class_name}" for class_name in
                       map(lambda x: ', '.join(x[:2]).lower(), dataset_test.classes)]
    elif dataset['name'] == "CUB":
        dataset_train = CUBDataset(dataset["root_dir"], train=True,
                                   transform=transform)
        dataset_test = CUBDataset(dataset["root_dir"], train=False,
                                  transform=transform)
        # TODO
        class_names = list(map(str, range(200)))

        caption_class_position = 0
    elif dataset['name'] == "HouseNumbers":
        dataset_train = HouseNumbersDataset(dataset['root_dir'], train=True, transform=transform)
        dataset_test = HouseNumbersDataset(dataset['root_dir'], train=False, transform=transform)
        class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        class_names = [f"the number {classname}" for classname in class_names]
        caption_class_position = 2
    else:
        raise ValueError(f"{dataset['name']} is not a valid dataset name.")

    return dataset_train, dataset_test, class_names, caption_class_position


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


def save_results(results_path, accuracies, confusion_matrices, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir()
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


def load_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    accuracies = np.load(results_path / "accuracies.npy", allow_pickle=True).item()
    confusion_matrices = np.load(results_path / "confusion_matrices.npy", allow_pickle=True).item()
    return accuracies, confusion_matrices, config


def load_corr_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    bert_corr = np.load(results_path / "bert_corr.npy", allow_pickle=True).item()
    resnet_corr = np.load(results_path / "resnet_corr.npy", allow_pickle=True).item()
    resnet_bert_score = np.load(results_path / "resnet_bert_score.npy", allow_pickle=True).item()
    return bert_corr, resnet_corr, resnet_bert_score, config
