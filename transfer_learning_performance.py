import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datasets.datasets import get_dataset
from utils.models import get_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    checkpoint = np.load(results_path / "checkpoint.npy", allow_pickle=True).item()
    return checkpoint, config


def save_results(results_path, checkpoint, config):
    print(f"Saving results in {str(results_path)}...")
    results_path.mkdir(exist_ok=True)
    np.save(str(results_path / "checkpoint.npy"), checkpoint)
    with open(str(results_path / "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)


def train_one_epoch(dataloader, model, probe, optimizer, device, running_average=1000):
    probe.train()
    losses = []
    running_loss = []
    for step, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, targets = images.to(device), targets.to(device)
        features = model.encode_image(images).float()
        prediction = probe(features.detach())
        loss = F.cross_entropy(prediction, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.detach().cpu().item())
        if step % running_average:
            losses.append(np.mean(running_loss))
            running_loss = []
    losses.append(np.mean(running_loss))
    return losses


def val(dataloader, model, probe, device):
    probe.eval()
    losses = []
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for step, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, targets = images.to(device), targets.to(device)
            features = model.encode_image(images).float()
            prediction = probe(features)
            loss = F.cross_entropy(prediction, targets)
            predicted = torch.argmax(prediction, dim=1)
            num_correct += (predicted == targets).sum().item()
            num_total += targets.size(0)
            losses.append(loss.detach().cpu().item())
    return np.mean(losses), num_correct / num_total


def train(dataset_train, dataset_val, model, probe, optimizer, scheduler, device, batch_size=64, num_epochs=100, num_workers=-1,
          **kwargs):
    dataloader_train = DataLoader(dataset_train, batch_size, num_workers=num_workers, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size, num_workers=num_workers)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_losses.extend(train_one_epoch(dataloader_train, model, probe, optimizer, device, **kwargs))
        val_loss, val_acc = val(dataloader_val, model, probe, device)
        scheduler.step()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    return train_losses, val_losses, val_accuracies


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    load_results_id = 237

    # Models to test
    model_names = [
        "CLIP-RN50",
        "virtex",
        "BiT-M-R50x1",
        "RN50",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "madry-imagenet_l2_3_0",
        # "CLIP-ViT-B/32",
        "geirhos-resnet50_trained_on_SIN",
        "semi-supervised-YFCC100M",
        "semi-weakly-supervised-instagram",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
    ]
    # Dataset to test on
    datasets = [
        {"name": "Birdsnap", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/birdsnap/birdsnap")},
        {"name": "Caltech101", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/caltech101/101_ObjectCategories")},
        {"name": "Caltech256", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/caltech256/256_ObjectCategories")},
        {"name": "DTD", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/DescribableTextures/dtd")},
        {"name": "FGVC-Aircraft", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/FGVC-Aircraft/fgvc-aircraft-2013b")},
        {"name": "Food101", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/food101/food-101")},
        {"name": "Flowers102", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/flowers102")},
        {"name": "IIITPets", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/OxfordPets")},
        {"name": "SUN397", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/SUN")},
        {"name": "StanfordCars", "batch_size": 80, "root_dir": os.path.expanduser("/mnt/SSD/datasets/StanfordCars")},
        {"name": "CIFAR10", "batch_size": 80, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR100", "batch_size": 80, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/StreetViewHouseNumbers/format2"},
        # {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/HD1/datasets/CUB/CUB_200_2011"},
        # {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        # {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
    ]
    # Number of prototypes per class and number of trials for each number of prototype
    prototypes_trials = {n_proto: 10 for n_proto in [1, 5, 10]}

    lr = 1e-3
    n_epochs = 20
    n_workers = 8

    plot_images = False

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
        "prototypes_trials": prototypes_trials
    }

    if load_results_id is not None:
        data, loaded_config = load_results(Path(f"results/{load_results_id}"))
    else:
        data = {
            "models": {},
            "train_losses": {},
            "val_losses": {},
            "val_acc": {}
        }

    items_to_remove = [
        # "CLIP-RN50",
    ]
    for model in items_to_remove:
        for item in data.keys():
            if model in data[item]:
                del data[item][model]

    try:
        for model_name in model_names:
            for item in data.keys():
                if model_name not in data[item]:
                    data[item][model_name] = {}
            # Import model
            model, transform = get_model(model_name, device)
            model.eval()

            for dataset in datasets:
                print(f"Model {model_name}. Dataset {dataset['name']}.")
                skipping = False
                for item in data.keys():
                    if dataset['name'] in data[item][model_name] and data[item][model_name][dataset['name']] != {}:
                        skipping = True
                        break
                if skipping:
                    print("Skipping.")
                    continue

                # Get dataset
                dataset_train, dataset_test, class_names, _ = get_dataset(dataset, transform, data_augment=True)

                linear_probe = torch.nn.Linear(model.out_dim, len(class_names)).to(device)
                optimizer = torch.optim.Adam(linear_probe.parameters(), lr=lr, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.1)

                train_losses, val_losses, val_acc = train(
                    dataset_train, dataset_test, model, linear_probe,
                    optimizer, scheduler, device, dataset['batch_size'], n_epochs,
                    n_workers
                )

                data['models'][model_name][dataset['name']] = linear_probe.state_dict()
                data['train_losses'][model_name][dataset['name']] = train_losses
                data['val_losses'][model_name][dataset['name']] = val_losses
                data['val_acc'][model_name][dataset['name']] = val_acc

        save_results(results_path, data, config)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(results_path, data, config)
        raise e
