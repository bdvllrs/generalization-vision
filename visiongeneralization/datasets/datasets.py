import os
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.datasets
from PIL import Image as Image
from natsort import natsorted
from scipy.io import loadmat
from torchvision import transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageNet

from visiongeneralization.datasets.aircraft import FGVCAircraft

Image.LOAD_TRUNCATED_IMAGES = True


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
        return image, int(label)


class Birdsnap(torchvision.datasets.ImageFolder):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "download/images"
        self.test_elements = self.root_dir / "test_images.txt"
        self.classes_file = self.root_dir / "species.txt"
        with open(self.test_elements, "r") as f:
            self.test_images = [line.rstrip('\n') for line in f][1:]

        with open(self.classes_file, "r") as f:
            self.classes = [line.rstrip('\n').split("\t")[1] for line in f][1:]

        def is_valid_file(path):
            if train:
                return '/'.join(path.split('/')[-2:]) not in self.test_images
            else:
                return '/'.join(path.split('/')[-2:]) in self.test_images

        super(Birdsnap, self).__init__(str(self.image_dir), transform, is_valid_file=is_valid_file)


class Food101(torchvision.datasets.ImageFolder):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.train_elements = self.root_dir / "meta/train.txt"
        self.test_elements = self.root_dir / "meta/test.txt"
        self.classes_file = self.root_dir / "meta/classes.txt"

        with open(self.train_elements, "r") as f:
            self.train_images = [line.rstrip('\n') + ".jpg" for line in f]

        with open(self.test_elements, "r") as f:
            self.test_images = [line.rstrip('\n') + ".jpg" for line in f]

        with open(self.classes_file, "r") as f:
            self.class_names = [line.replace("_", " ").rstrip('\n') for line in f]

        def is_valid_file(path):
            if train:
                return '/'.join(path.split('/')[-2:]) in self.train_images
            else:
                return '/'.join(path.split('/')[-2:]) in self.test_images

        super(Food101, self).__init__(str(self.image_dir), transform, is_valid_file=is_valid_file)


class DTD(torchvision.datasets.ImageFolder):
    def __init__(self, root_dir, split=1, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.train_path = self.root_dir / f"labels/train{split}.txt"
        self.val_path = self.root_dir / f"labels/val{split}.txt"
        self.test_path = self.root_dir / f"labels/test{split}.txt"

        if train:
            with open(self.train_path, "r") as f:
                self.images = [line.rstrip('\n') for line in f]
            with open(self.val_path, "r") as f:
                self.images += [line.rstrip('\n') for line in f]
        else:
            with open(self.test_path, "r") as f:
                self.images = [line.rstrip('\n') for line in f]

        def is_valid_file(path):
            if train:
                return '/'.join(path.split('/')[-2:]) not in self.images
            else:
                return '/'.join(path.split('/')[-2:]) in self.images

        super(DTD, self).__init__(str(self.image_dir), transform, is_valid_file=is_valid_file)


class Flowers102:
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "jpg"
        self.labels_dir = self.root_dir / "imagelabels.mat"
        self.set_id_dir = self.root_dir / "setid.mat"
        self.labels = loadmat(str(self.labels_dir))['labels'][0]
        self.set_id = loadmat(str(self.set_id_dir))
        self.trainval_id = self.set_id['trnid'].tolist()[0] + self.set_id['valid'].tolist()[0]
        self.test_id = self.set_id['tstid'].tolist()[0]

        self.samples = []
        for root, _, fnames in sorted(os.walk(self.image_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                index = fname[-9:-4]
                k = int(index) - 1
                if (train and k + 1 in self.trainval_id) or (not train and k + 1 in self.test_id):
                    item = path, int(self.labels[k]) - 1
                    self.samples.append(item)

        # TODO
        self.class_names = [f"{k}" for k in range(len(np.unique(self.labels)))]

        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class IIITPets:
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.train_val_dir = self.root_dir / "annotations/trainval.txt"
        self.test_dir = self.root_dir / "annotations/test.txt"

        self.class_names = ["" for k in range(37)]
        image_dir = self.train_val_dir if train else self.test_dir
        with open(image_dir, "r") as f:
            self.samples = []
            for line in f:
                fname = line.rstrip('\n').split(" ")[0]
                clsname = " ".join(fname.split("_")[:-1])
                image = str(self.image_dir / (fname + ".jpg"))
                label = int(line.rstrip('\n').split(" ")[1]) - 1
                self.samples.append((image, label))

                if not self.class_names[label]:
                    self.class_names[label] = clsname.lower()

        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class SUN397:
    def __init__(self, root_dir, split="01", train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "SUN397"
        if train:
            self.files = self.root_dir / f"Training_{split}.txt"
        else:
            self.files = self.root_dir / f"Testing_{split}.txt"
        self.class_file = self.root_dir / "ClassName.txt"

        with open(self.class_file, "r") as f:
            self.classes = {line.rstrip('\n'): k for k, line in enumerate(f)}
        with open(self.class_file, "r") as f:
            # TODO
            self.class_names = [line.rstrip('\n') for line in f]

        with open(self.files, "r") as f:
            self.samples = []
            for line in f:
                fname = line.rstrip('\n')
                clsname = "/".join(fname.split("/")[:-1])
                image = str(self.image_dir / fname[1:])
                label = self.classes[clsname]
                self.samples.append((image, label))

        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class StanfordCars:
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "car_ims"
        self.test_annos_path = self.root_dir / "cars_test_annos_withlabels.mat"
        self.train_annos_path = self.root_dir / "devkit/cars_train_annos.mat"
        self.meta_path = self.root_dir / "devkit/cars_meta.mat"

        self.test_annos = loadmat(str(self.test_annos_path))['annotations']
        self.train_annos = loadmat(str(self.train_annos_path))['annotations']
        self.class_names = [cls[0] for cls in loadmat(str(self.meta_path))['class_names'][0]]

        annos = self.train_annos if train else self.test_annos

        self.samples = []
        for item in annos[0]:
            bxm, bxM, bym, byM, cls, fname = item
            fname = fname[0]
            if len(fname) == 9:
                fname = "0" + fname

            image = str(self.image_dir / fname)
            label = cls[0, 0] - 1
            assert label >= 0
            self.samples.append((image, label))

        self.transform = transform
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class ImageNetVal150(torchvision.datasets.ImageFolder):
    """
        The class provides a Dataset method to get all the images in the 'main_dir' without their targets
    """
    cls2idx = {'tomato': 142, 'watermelon': 144, 'yarmulke, yarmulka, yarmelke': 141, 'lawn mower, mower': 88,
               'menorah': 92, 'tricycle, trike, velocipede': 136, 'camera tripod': 51, 'elephant': 37,
               'fire engine, fire truck': 64,
               'centipede': 9, 'pitcher, ewer': 102, 'baseball bat, lumber': 42, 'gorilla, Gorilla gorilla': 35,
               'speedboat': 121, 'toaster': 133,
               'shirt': 115, 'light bulb, lightbulb, bulb, incandescent lamp, electric light, electric-light bulb': 89,
               'soda can': 119,
               'lathe': 87, 'pool table, billiard table, snooker table': 103, 'trilobite': 6,
               'basket, basketball hoop, hoop': 44, 'cereal box': 53,
               'penguin': 19, 'spoon': 122, 'backpack, back pack, knapsack, packsack, rucksack, haversack': 41,
               'screwdriver': 112, 'tennis ball': 129,
               'horse, Equus caballus': 29, 'ghetto blaster, boom box': 70, 'dolphin': 20, 'snail': 14,
               'flashlight, torch': 65,
               'airship, dirigible': 39, 'porcupine, hedgehog': 28, 'birdbath': 48, 'triceratops': 4, 'hourglass': 80,
               'tepee, tipi, teepee': 131,
               'telephone booth, phone booth, call box, telephone box, telephone kiosk': 128,
               'stirrup, stirrup iron': 124,
               'refrigerator, icebox': 106, 'school bus': 111, 'gym shoe, sneaker, tennis shoe': 73,
               'horseshoe crab, king crab, Limulus polyphemus, Xiphosurus polyphemus': 10,
               'computer monitor': 58, 'soccer ball': 118, 'diskette, floppy, floppy disk': 60,
               'marimba, xylophone': 90,
               'paper clip, paperclip, gem clip': 99, 'skateboard': 116, 'true toad': 2,
               'chimpanzee, chimp, Pan troglodytes': 36,
               'microscope': 93, 'bonsai': 149, 'helicopter, chopper, whirlybird, eggbeater': 77,
               'harmonica, mouth organ, harp, mouth harp': 75,
               'car wheel': 52, 'common raccoon, common racoon, coon, ringtail, Procyon lotor': 38,
               'giraffe, camelopard, Giraffa camelopardalis': 32,
               'sunflower, helianthus': 146, 'necktie, tie': 97, 'laptop, laptop computer': 86, 'goose': 12,
               'harpsichord, cembalo': 76, 'joystick': 81,
               'miniature fan palm, bamboo palm, fern rhapis, Rhapis excelsa': 148, 'watch, ticker': 138,
               'personal digital assistant, PDA, personal organizer, personal organiser, organizer, organiser': 100,
               'snake, serpent, ophidian': 5,
               'bathtub, bathing tub, bath, tub': 45, 'grape': 145, 'scorpion': 7, 'ketch': 84,
               'spectacles, specs, eyeglasses, glasses': 120, 'zebra': 30,
               'dog, domestic dog, Canis familiaris': 21, 'steering wheel, wheel': 123, 'hummingbird': 11,
               'skunk, polecat, wood pussy': 33, 'obelisk': 98,
               'revolver, six-gun, six-shooter': 107, 'syringe': 126, 'tennis racket, tennis racquet': 130,
               'bowling pin, pin': 49, 'French horn, horn': 66,
               'kayak': 83, 'boxing glove, glove': 50, 'palm, palm tree': 147, 'windmill': 139,
               'frog, toad, toad frog, anuran, batrachian, salientian': 1, 'golf ball': 71,
               'Frisbee': 67, 'treadmill': 135, 'coffee mug': 55, 'fighter, fighter aircraft, attack aircraft': 63,
               'Segway, Segway Human Transporter, Segway HT': 113,
               'grasshopper, hopper': 24, 'knife': 85, 'kangaroo': 13, 'dumbbell': 61, 'ostrich, Struthio camelus': 0,
               'beacon, lighthouse, beacon light, pharos': 46,
               'theodolite, transit': 132, 'cormorant, Phalacrocorax carbo': 18, 'bear': 23, 'skyscraper': 117,
               'radio telescope, radio reflector': 105, 'octopus, devilfish': 16,
               'greyhound': 22, 'elk, European elk, moose, Alces alces': 31, 'projector': 104,
               'sword, blade, brand, steel': 125, 'hot-air balloon': 78, 'wine bottle': 140,
               'computer keyboard, keypad': 57,
               'baby buggy, baby carriage, carriage, perambulator, pram, stroller, go-cart, pushchair, pusher': 40,
               'clasp knife, jackknife': 54, 'mountain bike, all-terrain bike, off-roader': 95, 'tuning fork': 137,
               'hot tub': 79, 'Kalashnikov': 82, 'spider': 8,
               'ibis': 17, 'earphone, earpiece, headphone, phone': 62, 'hand calculator, pocket calculator': 74,
               'megaphone': 91, 'mushroom': 143, 'mouse, computer mouse': 96,
               'rifle': 108, 'photocopier': 101, 'homo, man, human being, human': 34, 'motorcycle, bike': 94,
               'compact disk, compact disc, CD': 56,
               'baseball glove, glove, baseball mitt, mitt': 43, 'praying mantis, praying mantid, Mantis religioso': 26,
               'cockroach, roach': 25, 'starfish, sea star': 27,
               'mussel': 15, 'roulette wheel, wheel': 109, 'sextant': 114,
               'binoculars, field glasses, opera glasses': 47, 'toaster oven': 134, 'guitar pick': 72,
               'hawksbill turtle, hawksbill, hawkbill, tortoiseshell turtle, Eretmochelys imbricata': 3,
               'dial telephone, dial phone': 59, 'teapot': 127,
               'gas pump, gasoline pump, petrol pump, island dispenser': 69, 'saddle': 110,
               'frying pan, frypan, skillet': 68}

    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        super(ImageNetVal150, self).__init__(str(self.image_dir), transform)


def get_dataset(dataset, transform, data_augment=False):
    caption_class_position = 1
    if dataset['name'] == "MNIST":
        # Download the dataset
        dataset_train = MNIST(root=dataset["root_dir"], download=True,
                              train=True, transform=transform(32, data_augment))
        dataset_test = MNIST(root=dataset["root_dir"], download=True,
                             train=False, transform=transform(32, False))
        class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        class_names = [f"the number {classname}" for classname in class_names]
        caption_class_position = 2
    elif dataset['name'] == "FashionMNIST":
        dataset_train = FashionMNIST(root=dataset["root_dir"], download=True,
                                     train=True, transform=transform(32, data_augment))
        dataset_test = FashionMNIST(root=dataset["root_dir"], download=True,
                                    train=False, transform=transform(32, False))
        class_names = [f"a {class_name}" for class_name in map(lambda x: x.lower(), dataset_test.classes)]
    elif dataset['name'] == "CIFAR10":
        dataset_train = CIFAR10(root=dataset["root_dir"], download=True,
                                train=True, transform=transform(32, data_augment))
        dataset_test = CIFAR10(root=dataset["root_dir"], download=True,
                               train=False, transform=transform(32, False))
        class_names = [f"a {class_name}" for class_name in map(lambda x: x.lower(), dataset_test.classes)]
    elif dataset['name'] == "CIFAR100":
        dataset_train = CIFAR100(root=dataset["root_dir"], download=True,
                                 train=True, transform=transform(32, data_augment))
        dataset_test = CIFAR100(root=dataset["root_dir"], download=True,
                                train=False, transform=transform(32, False))
        class_names = [f"a {class_name}" for class_name in map(lambda x: x.lower(), dataset_test.classes)]
    elif dataset['name'] == "ImageNet":
        dataset_train = ImageNet(root=dataset["root_dir"], split='train',
                                 transform=transform(256, data_augment))
        dataset_test = ImageNet(root=dataset["root_dir"], split='val',
                                transform=transform(256, False))
        class_names = [f"a {class_name}" for class_name in
                       map(lambda x: ', '.join(x[:2]).lower(), dataset_test.classes)]
    elif dataset['name'] == "ImageNetVal150":
        dataset_train = None
        dataset_test = ImageNetVal150(dataset["root_dir"], transform(256, False))
        class_names = [f"a {class_name}" for class_name in dataset_test.classes]
    elif dataset['name'] == "CUB":
        dataset_train = CUBDataset(dataset["root_dir"], train=True,
                                   transform=transform(256, data_augment))
        dataset_test = CUBDataset(dataset["root_dir"], train=False,
                                  transform=transform(256, False))
        # TODO
        class_names = list(map(str, range(200)))

        caption_class_position = 0
    elif dataset['name'] == "HouseNumbers":
        dataset_train = HouseNumbersDataset(dataset['root_dir'], train=True, transform=transform(32, data_augment))
        dataset_test = HouseNumbersDataset(dataset['root_dir'], train=False, transform=transform(32, False))
        class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        class_names = [f"the number {classname}" for classname in class_names]
        caption_class_position = 2
    elif dataset['name'] == "Birdsnap":
        dataset_train = Birdsnap(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = Birdsnap(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.classes
        class_names = [f"the bird {classname}" for classname in class_names]
        caption_class_position = 2
    elif dataset['name'] in ["Caltech101", "Caltech256"]:
        caltech_train = torchvision.datasets.ImageFolder(dataset['root_dir'], transform(256, data_augment))
        caltech_test = torchvision.datasets.ImageFolder(dataset['root_dir'], transform(256, False))
        NUM_TRAINING_SAMPLES_PER_CLASS = 30 if dataset['name'] == "Caltech101" else 60
        class_start_idx = [0] + [i for i in np.arange(1, len(caltech_train)) if
                                 caltech_train.targets[i] == caltech_train.targets[i - 1] + 1]
        train_indices = sum([np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in
                             class_start_idx], [])
        test_indices = list((set(np.arange(1, len(caltech_train))) - set(train_indices)))
        dataset_train = torch.utils.data.Subset(caltech_train, train_indices)
        dataset_test = torch.utils.data.Subset(caltech_test, test_indices)
        if dataset['name'] == "Caltech101":
            class_names = list(map(lambda cls: cls.replace("_", " "), dataset_train.dataset.classes))
        else:
            class_names = list(map(lambda cls: cls.split('.')[1].replace("-", " "), dataset_train.dataset.classes))
        caption_class_position = 0
    elif dataset['name'] == "DTD":
        dataset_train = DTD(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = DTD(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.classes
        class_names = [f"a {classname} texture" for classname in class_names]
        caption_class_position = 1
    elif dataset['name'] == "FGVC-Aircraft":
        dataset_train = FGVCAircraft(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = FGVCAircraft(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = [cls.rstrip("\n") for cls in dataset_train.classes]
        class_names = [f"the {classname} aircraft" for classname in class_names]
        caption_class_position = 1
    elif dataset['name'] == "Food101":
        dataset_train = Food101(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = Food101(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.class_names
        class_names = [f"a {classname}" for classname in class_names]
        caption_class_position = 1
    elif dataset['name'] == "Flowers102":
        dataset_train = Flowers102(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = Flowers102(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.class_names
        class_names = [f"class {classname}" for classname in class_names]
        caption_class_position = 1
    elif dataset['name'] == "IIITPets":
        dataset_train = IIITPets(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = IIITPets(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.class_names
        class_names = [f"a {classname}" for classname in class_names]
        caption_class_position = 1
    elif dataset['name'] == "SUN397":
        dataset_train = SUN397(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = SUN397(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.class_names
        class_names = [f"a {classname}" for classname in class_names]
        caption_class_position = 1
    elif dataset['name'] == "StanfordCars":
        dataset_train = StanfordCars(dataset['root_dir'], train=True, transform=transform(256, data_augment))
        dataset_test = StanfordCars(dataset['root_dir'], train=False, transform=transform(256, False))
        class_names = dataset_train.class_names
        class_names = [f"a {classname}" for classname in class_names]
        caption_class_position = 1
    else:
        raise ValueError(f"{dataset['name']} is not a valid dataset name.")
    return dataset_train, dataset_test, class_names, caption_class_position
