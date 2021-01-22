import json
import os
from pathlib import Path

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from robustness.imagenet_models import resnet50
from sklearn.metrics import confusion_matrix as confusion_matrix_, accuracy_score as accuracy_score_
from torch.utils import model_zoo
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision.models import resnet50
from tqdm import tqdm

import clip

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

    def encode_image(self, images):
        return self.module(images)


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
        label = self.labels[file_idx] - 1 # Start index at 0
        with Image.open(self.root_dir / "images" / self.location[file_idx]) as image:
            processed_image = self.transform(image)
        return processed_image, label


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


def get_prototypes(model_, train_set, device, n_examples_per_class=5, n_classes=10):
    assert n_examples_per_class >= 1

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
    for proto_imgs in prototypes:
        imgs = torch.stack(proto_imgs, dim=0).to(device)
        feature = model_.encode_image(imgs).mean(dim=0)
        feature /= feature.norm(dim=-1, keepdim=True)
        features.append(feature)
    return torch.stack(features, dim=0)


def get_model(model_name, device):
    if "CLIP" in model_name and model_name.replace("CLIP-", "") in clip_models:
        model, transform = clip.load(model_name.replace("CLIP-", ""), device=device, jit=False)
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
    else:
        raise ValueError(f"{model_name} is not a valid model name.")
    return model, transform


def get_dataset(dataset, transform):
    if dataset['name'] == "MNIST":
        # Download the dataset
        dataset_train = MNIST(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = MNIST(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = list(map(str, range(10)))
    elif dataset['name'] == "FashionMNIST":
        dataset_train = FashionMNIST(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = FashionMNIST(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = list(map(str, range(10)))
    elif dataset['name'] == "CIFAR10":
        dataset_train = CIFAR10(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = CIFAR10(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = list(map(str, range(10)))
    elif dataset['name'] == "CIFAR100":
        dataset_train = CIFAR100(root=dataset["root_dir"], download=True, train=True, transform=transform)
        dataset_test = CIFAR100(root=dataset["root_dir"], download=True, train=False, transform=transform)
        class_names = list(map(str, range(100)))
    elif dataset['name'] == "CUB":
        dataset_train = CUBDataset(dataset["root_dir"], train=True,
                                   transform=transform)
        dataset_test = CUBDataset(dataset["root_dir"], train=False,
                                  transform=transform)
        class_names = list(map(str, range(200)))
    else:
        raise ValueError(f"{dataset['name']} is not a valid dataset name.")

    return dataset_train, dataset_test, class_names


def save_results(results_path, accuracies, confusion_matrices, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir()
    np.save(str(results_path / "accuracies.npy"), accuracies)
    np.save(str(results_path / "confusion_matrices.npy"), confusion_matrices)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


def load_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    accuracies = np.load(results_path / "accuracies.npy", allow_pickle=True).item()
    confusion_matrices = np.load(results_path / "confusion_matrices.npy", allow_pickle=True).item()
    return accuracies, confusion_matrices, config
