import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from visiongeneralization.datasets.datasets import get_dataset
from visiongeneralization.models import get_model
from visiongeneralization.utils import run, save_results

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def train(dataset_train, dataset_val, model, probe, optimizer, scheduler, device, batch_size=64, num_epochs=100,
          num_workers=-1,
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


def main(config, checkpoint):
    try:
        model_names, datasets = config["model_names"], config["datasets"]
        override_models = config["override_models"]
        lr, n_epochs, n_workers = config["lr"], config["n_epochs"], config["n_workers"]

        for model in override_models:
            for item in checkpoint.keys():
                if model in checkpoint[item]:
                    del checkpoint[item][model]

        # prepare dictionaries by setting them empty
        for model_name in model_names:
            for item in checkpoint.keys():
                if model_name not in checkpoint[item]:
                    checkpoint[item][model_name] = {}

            # Import model
            model, transform = get_model(model_name, device)
            model.eval()

            for dataset in datasets:
                print(f"Model {model_name}. Dataset {dataset['name']}.")
                skipping = False
                # check that data has not been already computed
                for item in checkpoint.keys():
                    if dataset['name'] in checkpoint[item][model_name] and checkpoint[item][model_name][
                        dataset['name']] != {}:
                        skipping = True
                        break
                if skipping:
                    print("Skipping.")
                    continue

                # Get dataset
                dataset_train, dataset_test, class_names, _ = get_dataset(dataset, transform, data_augment=True)

                # Define net parameters, optimizer, learning schedule
                linear_probe = torch.nn.Linear(model.out_dim, len(class_names)).to(device)
                optimizer = torch.optim.Adam(linear_probe.parameters(), lr=lr, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.1)

                # train the linear probe
                train_losses, val_losses, val_acc = train(
                    dataset_train, dataset_test, model, linear_probe,
                    optimizer, scheduler, device, dataset['batch_size'], n_epochs,
                    n_workers
                )

                print("Saving...")
                checkpoint['models'][model_name][dataset['name']] = linear_probe.state_dict()
                checkpoint['train_losses'][model_name][dataset['name']] = train_losses
                checkpoint['val_losses'][model_name][dataset['name']] = val_losses
                checkpoint['val_acc'][model_name][dataset['name']] = val_acc
                save_results(config["results_path"], config, checkpoint=checkpoint)

        save_results(config["results_path"], config, checkpoint=checkpoint)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(config["results_path"], config, checkpoint=checkpoint)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot generalization task.')
    parser.add_argument('--load_results', default=None, type=int,
                        help='Id of a previous experiment to continue.')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate.')
    parser.add_argument('--n_epochs', default=150, type=int,
                        help='Number of epochs.')
    parser.add_argument('--n_workers', default=0, type=int,
                        help='Number of workers.')
    parser.add_argument('--batch_size', default=80, type=int,
                        help='Batch size.')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # "semi-supervised-YFCC100M",
        # "semi-weakly-supervised-instagram",
        "ICMLM",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        # "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
    ]

    # Dataset to test on
    datasets = [
        # {"name": "Birdsnap", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/birdsnap/birdsnap")},
        # {"name": "Caltech101", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/caltech101/101_ObjectCategories")},
        # {"name": "Caltech256", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/caltech256/256_ObjectCategories")},
        # {"name": "DTD", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/DescribableTextures/dtd")},
        # {"name": "FGVC-Aircraft", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/FGVC-Aircraft/fgvc-aircraft-2013b")},
        # {"name": "Food101", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/food101/food-101")},
        # {"name": "Flowers102", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/flowers102")},
        # {"name": "IIITPets", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/OxfordPets")},
        # {"name": "SUN397", "batch_size": args.batch_size, "root_dir": os.path.expanduser("/mnt/SSD/datasets/SUN")},
        # {"name": "StanfordCars", "batch_size": args.batch_size,
        #  "root_dir": os.path.expanduser("/mnt/SSD/datasets/StanfordCars")},
        {"name": "CIFAR10", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "CIFAR100", "batch_size": args.batch_size, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "HouseNumbers", "batch_size": 64, "root_dir": "/mnt/SSD/datasets/StreetViewHouseNumbers/format2"},
        {"name": "CUB", "batch_size": 64, "root_dir": "/mnt/SSD/datasets/CUB/CUB_200_2011"},
        {"name": "MNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
        {"name": "FashionMNIST", "batch_size": 64, "root_dir": os.path.expanduser("~/.cache")},
    ]

    config = {
        "model_names": model_names,
        "datasets": datasets,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "n_workers": args.n_workers,
        "override_models": []
    }

    checkpoint = {
        "models": {},
        "train_losses": {},
        "val_losses": {},
        "val_acc": {}
    }

    run(main, config, args.load_results, checkpoint=checkpoint)
