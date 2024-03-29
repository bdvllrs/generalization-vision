import argparse
import os

import numpy as np
import torch
from PIL.ImageFile import ImageFile
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from visiongeneralization.datasets.datasets import get_dataset
from visiongeneralization.models import get_model
from visiongeneralization.utils import run, save_results, max_margin_loss, load_conf, available_model_names

ImageFile.LOAD_TRUNCATED_IMAGES = True


def filter_classes(dataset, class_labels):
    items = [k for k, (_, label) in enumerate(dataset) if label in class_labels]
    return Subset(dataset, items)


def train_one_epoch(dataloader, model, lm_features, vision_to_text, optimizer, device,
                    running_average=1000):
    vision_to_text.train()

    losses = []
    running_loss = []
    for step, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images, targets = images.to(device), targets.to(device)
        # project vision features to language features
        features_i = vision_to_text(model.encode_image(images).float())
        # compute compatibility
        scores = features_i @ lm_features.transpose(1, 0)
        predictions = torch.softmax(scores, dim=-1)
        loss = max_margin_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.detach().cpu().item())
        if step % running_average:
            losses.append(np.mean(running_loss))
            running_loss = []
    losses.append(np.mean(running_loss))
    return losses


def val(dataloader, model, lm_features, vision_to_text, device, seen_labels):
    vision_to_text.eval()
    losses = []
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for step, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images, targets = images.to(device), targets.to(device)
            features_i = vision_to_text(model.encode_image(images).float())
            # compute compatibility
            scores = features_i @ lm_features.transpose(1, 0)
            scores[:, seen_labels] = - torch.tensor(10000).float().to(device)
            predictions = torch.softmax(scores, dim=-1)
            loss = max_margin_loss(predictions, targets)

            predicted = torch.argmax(scores, dim=1)
            num_correct += (predicted == targets).sum().item()
            num_total += targets.size(0)
            losses.append(loss.detach().cpu().item())
    return np.mean(losses), num_correct / num_total


def train(dataset_train, dataset_val, model, lm_features, vision_to_text, optimizer, scheduler, device,
          seen_labels,
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
            train_one_epoch(dataloader_train, model, lm_features, vision_to_text, optimizer, device,
                            **kwargs))
        val_loss, val_acc = val(dataloader_val, model, lm_features, vision_to_text, device, seen_labels)
        print(f"Val acc: {val_acc}")
        scheduler.step()
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    return train_losses, val_losses, val_accuracies


def main(config, checkpoint):
    try:
        model_names, datasets = config["model_names"], config["datasets"]
        override_models = config["override_models"]
        lr, n_epochs, n_workers = config["lr"], config["n_epochs"], config["n_workers"]
        weight_decay = config["weight_decay"]
        step_lr_schedule = config["step_lr_schedule"]
        gamma_lr_schedule = config["gamma_lr_schedule"]
        n_trials = config["n_trials"]
        seen_prop = config["seen_prop"]
        lm_model_name = config["lm"]
        caption_sentence_prototypes = config["caption_sentence_prototypes"]

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
            assert model.has_image_encoder
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
                dataset_train, dataset_test, class_names, caption_class_location = get_dataset(dataset, transform,
                                                                                               data_augment=True)

                # Get text vectors for the classes
                caption_prototype, class_token_position = caption_sentence_prototypes
                # Add the classnames to the captions
                captions = [caption_prototype.format(classname=classname) for classname in class_names]
                with torch.no_grad():
                    inputs = tokenizer(captions)
                    language_features = lm_model.encode_text(inputs, device,
                                                             class_token_position + caption_class_location)
                    language_features = language_features.to(device)

                n_seen_classes = int(seen_prop * len(class_names))
                print(f"{n_seen_classes} seen classes, {len(class_names) - n_seen_classes} unseen classes.")

                trial_history = {
                    "seen_labels": [],
                    "unseen_labels": [],
                    "models": [],
                    "train_losses": [],
                    "val_losses": [],
                    "val_acc": [],
                    "train_dataset_size": [],
                    "test_dataset_size": []
                }

                for trial in range(n_trials):
                    labels = np.arange(len(class_names))
                    np.random.shuffle(labels)
                    # Randomly choose the seen/unseen labels
                    seen_labels = sorted(labels[:n_seen_classes])
                    unseen_labels = sorted(labels[n_seen_classes:])

                    seen_dataset_train = filter_classes(dataset_train, seen_labels)
                    unseen_dataset_test = filter_classes(dataset_test, unseen_labels)

                    # Define net parameters, optimizer, learning schedule
                    vision_to_text = torch.nn.Linear(model.out_dim, lm_model.text_out_dim, bias=False).to(device)

                    optimizer = torch.optim.Adam(vision_to_text.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_lr_schedule, gamma_lr_schedule)

                    # train the linear probe
                    train_losses, val_losses, val_acc = train(
                        seen_dataset_train, unseen_dataset_test, model, language_features, vision_to_text,
                        optimizer, scheduler, device, seen_labels, dataset['batch_size'], n_epochs,
                        n_workers
                    )

                    trial_history['models'].append(vision_to_text.state_dict())
                    trial_history['train_losses'].append(train_losses)
                    trial_history['val_losses'].append(val_losses)
                    trial_history['val_acc'].append(val_acc)
                    trial_history['seen_labels'].append(seen_labels)
                    trial_history['unseen_labels'].append(unseen_labels)
                    trial_history['train_dataset_size'].append(len(seen_dataset_train))
                    trial_history['test_dataset_size'].append(len(unseen_dataset_test))

                checkpoint['models'][model_name][dataset['name']] = trial_history['models']
                checkpoint['train_losses'][model_name][dataset['name']] = trial_history['train_losses']
                checkpoint['val_losses'][model_name][dataset['name']] = trial_history['val_losses']
                checkpoint['val_acc'][model_name][dataset['name']] = trial_history['val_acc']
                checkpoint['seen_labels'][model_name][dataset['name']] = trial_history['seen_labels']
                checkpoint['unseen_labels'][model_name][dataset['name']] = trial_history['unseen_labels']
                checkpoint['train_dataset_size'][model_name][dataset['name']] = trial_history['train_dataset_size']
                checkpoint['test_dataset_size'][model_name][dataset['name']] = trial_history['test_dataset_size']
                print("Saving...")
                save_results(config["results_path"], config, checkpoint=checkpoint)

        save_results(config["results_path"], config, checkpoint=checkpoint)
    except BaseException as e:
        print("Something happened... Saving results so far.")
        save_results(config["results_path"], config, checkpoint=checkpoint)
        raise e


if __name__ == '__main__':
    conf = load_conf()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay.')
    parser.add_argument('--step_lr_schedule', default=10, type=int,
                        help='Step for learning rate schedule.')
    parser.add_argument('--gamma_lr_schedule', default=0.5, type=int,
                        help='Gamma for learning rate schedule.')
    parser.add_argument('--n_trials', default=10, type=int,
                        help='Number of trials for seen/unseen class selection.')
    parser.add_argument('--seen_prop', default=0.7, type=int,
                        help='Proportion of seen classes compared to unseen classes.')
    parser.add_argument('--n_epochs', default=20, type=int,
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
        "weight_decay": args.weight_decay,
        "step_lr_schedule": args.step_lr_schedule,
        "gamma_lr_schedule": args.gamma_lr_schedule,
        "n_epochs": args.n_epochs,
        "n_workers": args.n_workers,
        "lm": args.language_model,
        "n_trials": args.n_trials,
        "seen_prop": args.seen_prop,
        "caption_sentence_prototypes": conf.caption_sentence_prototype,
        "override_models": args.override_models
    }

    checkpoint = {
        "models": {},
        "train_losses": {},
        "val_losses": {},
        "val_acc": {},
        "seen_labels": {},
        "unseen_labels": {},
        "train_dataset_size": {},
        "test_dataset_size": {}
    }

    run(main, config, load_results_id, checkpoint=checkpoint)
