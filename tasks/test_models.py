import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from visiongeneralization.datasets.datasets import get_dataset
from visiongeneralization.models import get_model
from visiongeneralization.utils import run


def val(model_, dataset_, batch_size):
    dataloader = DataLoader(dataset_, batch_size)
    losses = []
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for step, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, targets = images.to(device), targets.to(device)
            prediction = model_.encode_image(images).float()
            loss = F.cross_entropy(prediction, targets)
            predicted = torch.argmax(prediction, dim=1)
            num_correct += (predicted == targets).sum().item()
            num_total += targets.size(0)
            losses.append(loss.detach().cpu().item())
    return np.mean(losses), num_correct / num_total


def main(config):
    model_names, datasets = config["model_names"], config["datasets"]

    with torch.no_grad():
        for model_name in model_names:
            print(model_name)
            # Import model
            model, transform, _ = get_model(model_name, device, keep_fc=True)
            model.eval()

            for dataset in datasets:
                # Get dataset
                print(dataset["name"])
                dataset_train, dataset_test, class_names, _ = get_dataset(dataset, transform)
                loss, acc = val(model, dataset_test, dataset["batch_size"])
                print("Val acc:", acc)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Test models on ImageNet task.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size.')
    args = parser.parse_args()

    batch_size = args.batch_size

    # Models to test
    model_names = [
        # "semi-supervised-YFCC100M",
        # "semi-weakly-supervised-instagram",
        # "BiT-M-R50x1",
        "RN50",
        "geirhos-resnet50_trained_on_SIN",
        "madry-imagenet_l2_3_0",
        # "virtex",
        "geirhos-resnet50_trained_on_SIN_and_IN",
        "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
        "madry-imagenet_linf_4",
        "madry-imagenet_linf_8",
        # "CLIP-ViT-B/32",
        # "CLIP-RN50",
    ]

    # Dataset to test on
    datasets = [
        {"name": "ImageNet", "batch_size": args.batch_size, "root_dir": "/mnt/SSD/datasets/imagenet"},
    ]

    config = {
        "model_names": model_names,
        "datasets": datasets
    }

    run(main, config, None)
