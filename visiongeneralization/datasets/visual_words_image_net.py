from torchvision.datasets import ImageNet
from nltk.corpus import wordnet as wn


class VisualWordsImagenet:
    def __init__(self, root, split="train", transform=None):
        self.dataset = ImageNet(root=root, split=split, transform=transform)

        self.label_map = []
        self.classes_to_idx = {}
        self.last_label = 0
        self.classes = []

        for k, classes in enumerate(self.dataset.classes):
            cls = classes[0]
            synset = wn.synsets(cls.replace(' ', '_'))[0]
            while len(cls.replace(" ", "_").split("_")) > 1:
                synset = synset.hypernyms()[0]
                cls = synset.lemmas()[0].name()
            if cls not in self.classes_to_idx:
                self.classes_to_idx[cls] = self.last_label
                self.classes.append(cls)
                self.last_label += 1
            self.label_map.append(self.classes_to_idx[cls])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return img, self.label_map[label]  # change label
