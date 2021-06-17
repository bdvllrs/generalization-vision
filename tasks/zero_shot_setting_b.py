import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL.ImageFile import ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm

from visiongeneralization.datasets.datasets import get_dataset
from visiongeneralization.models import get_model
from visiongeneralization.utils import run, save_results, load_conf, available_model_names

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_one_epoch(dataloader, model, lm_model, vision_to_text, text_to_vision, optimizer, device,
                    running_average=1000):
    vision_to_text.train()
    text_to_vision.train()

    losses = []
    running_loss = []
    for step, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, targets = images.to(device), targets.to(device)
        features_i = vision_to_text(model.encode_image(images).float())

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


def val(dataloader, model, lm_model, vision_to_text, text_to_vision, device):
    vision_to_text.eval()
    text_to_vision.eval()
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


def train(dataset_train, dataset_val, model, lm_model, vision_to_text, text_to_vision, optimizer, scheduler, device,
          batch_size=64, num_epochs=100,
          num_workers=-1,
          **kwargs):
    dataloader_train = DataLoader(dataset_train, batch_size, num_workers=num_workers, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size, num_workers=num_workers)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_losses.extend(
            train_one_epoch(dataloader_train, model, lm_model, vision_to_text, text_to_vision, optimizer, device,
                            **kwargs))
        val_loss, val_acc = val(dataloader_val, model, lm_model, vision_to_text, text_to_vision, device)
        scheduler.step()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    return train_losses, val_losses, val_accuracies


def main(config, checkpoint):
    try:
        model_names, datasets = config["model_names"], config["datasets"]
        override_models = config["override_models"]
        lr, n_epochs, n_workers = config["lr"], config["n_epochs"], config["n_workers"]
        z_size = config["embedding_size"]
        lm_model_name = config["lm"]

        # Remove some modes to recompute them.
        for model in override_models:
            for item in checkpoint.keys():
                if model in checkpoint[item]:
                    del checkpoint[item][model]

        # Import language model
        lm_model, _, tokenizer = get_model(lm_model_name, device)
        assert lm_model.has_text_encoder
        lm_model.eval()

        # prepare dictionaries by setting them empty
        for model_name in model_names:
            for item in checkpoint.keys():
                if model_name not in checkpoint[item]:
                    checkpoint[item][model_name] = {}

            # Import model
            model, transform, _ = get_model(model_name, device)
            assert model.has_vision_encoder
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
                vision_to_text = torch.nn.Linear(model.out_dim, z_size, bias=False).to(device)
                text_to_vision = torch.nn.Linear(lm_model_name.text_out_dim, z_size, bias=False).to(device)

                params = list(vision_to_text.parameters()) + list(text_to_vision.parameters())
                optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.1)

                # train the linear probe
                train_losses, val_losses, val_acc = train(
                    dataset_train, dataset_test, model, lm_model, vision_to_text, text_to_vision,
                    optimizer, scheduler, device, dataset['batch_size'], n_epochs,
                    n_workers
                )

                print("Saving...")
                checkpoint['models'][model_name][dataset['name']] = {"v2l": vision_to_text.state_dict(),
                                                                     "l2v": text_to_vision.state_dict()}
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = load_conf()

    parser = argparse.ArgumentParser(description='Zero-shot generalization task.')
    parser.add_argument('--load_results', default=None, type=int,
                        help='Id of a previous experiment to continue.')
    parser.add_argument('--models', type=str, nargs="+",
                        default=available_model_names(conf), choices=available_model_names(conf), help='Model to use.')
    parser.add_argument('--override_models', type=str, nargs="+",
                        default=[], choices=available_model_names(conf), help='Models to override.')
    parser.add_argument('--language_model', type=str, default="GPT2", choices=["GPT2", "BERT", "CLIP"],
                        help='Language model to use for the projection.')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate.')
    parser.add_argument('--embedding_size', default=256, type=int,
                        help='Embedding size of the multimodal space.')
    parser.add_argument('--n_epochs', default=150, type=int,
                        help='Number of epochs.')
    parser.add_argument('--n_workers', default=0, type=int,
                        help='Number of workers.')
    parser.add_argument('--batch_size', default=80, type=int,
                        help='Batch size.')
    args = parser.parse_args()

    load_results_id = args.load_results
    batch_size = args.batch_size

    # Models to test
    model_names = args.models

    # Dataset to test on
    datasets = [
        {"name": "CIFAR10", "batch_size": args.batch_size, "root_dir": conf.datasets.CIFAR10},
        {"name": "HouseNumbers", "batch_size": args.batch_size, "root_dir": conf.datasets.SVHN},
        {"name": "CUB", "batch_size": args.batch_size, "root_dir": conf.datasets.CUB},
        {"name": "CIFAR100", "batch_size": args.batch_size, "root_dir": conf.datasets.CIFAR100},
        {"name": "MNIST", "batch_size": args.batch_size, "root_dir": conf.datasets.MNIST},
        {"name": "FashionMNIST", "batch_size": args.batch_size, "root_dir": conf.datasets.FashionMNIST},
    ]

    config = {
        "model_names": model_names,
        "datasets": datasets,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "n_workers": args.n_workers,
        "lm": args.language_model,
        "embedding_size": args.embedding_size,
        "override_models": args.override_models
    }

    run(main, config, load_results_id, checkpoint={})
