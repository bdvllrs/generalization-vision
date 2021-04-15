import os
from pathlib import Path

import numpy as np
import torchvision.datasets
from PIL import Image
from scipy.io import loadmat

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
