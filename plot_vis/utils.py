from dataclasses import dataclass

import numpy as np

model_names_short = {
    "none": "Baseline (Skip-Gram)",
    "BERT": "BERT",
    "GPT2": "GPT2",
    "BERT_text": "BERT",
    "GPT2_text": "GPT2",
    "TSM-v": "TSM",
    "TSM-va": "TSM-va",
    "TSM-vat": "TSM-vat",
    "TSM-visual": "TSM",
    "TSM-shared": "TSM-vat",
    "GPV": "GPV",
    "GPV-SCE": "GPV-SCE",
    # "Word2Vec",
    "CLIP-RN50": "CLIP",
    "CLIP-RN50_text": "CLIP-T",
    "RN50": "RN50",
    "virtex": "VirTex",
    "ICMLM": "ICMLM",
    "BiT-M-R50x1": "BiT-M",
    "madry-imagenet_l2_3_0": "AR-L2",
    "madry-imagenet_linf_4": "AR-LI4",
    "madry-imagenet_linf_8": "AR-LI8",
    "geirhos-resnet50_trained_on_SIN": "SIN",
    "geirhos-resnet50_trained_on_SIN_and_IN": "SIN+IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": "SIN+IN-FIN",
    "semi-supervised-YFCC100M": "YFCC100M",
    "semi-weakly-supervised-instagram": "IG",
}

dataset_names_short = {
    "HouseNumbers": "SVHN"
}

color_scheme = {
    "BERT_text": "black",
    "GPT2_text": "black",
    # "Word2Vec",
    "CLIP-RN50": "xkcd:blue",
    "CLIP-RN50_text": "xkcd:blue",
    "virtex": "xkcd:blue",
    "ICMLM": "xkcd:blue",
    "TSM-v": "xkcd:blue",
    "GPV": "xkcd:blue",
    "GPV-SCE": "xkcd:blue",
    "TSM-vat": "xkcd:blue",
    "TSM-visual": "xkcd:blue",
    "TSM-shared": "xkcd:blue",
    "RN50": "xkcd:orange",
    "BiT-M-R50x1": "xkcd:puce",
    "madry-imagenet_l2_3_0": "xkcd:red",
    "madry-imagenet_linf_4": "xkcd:red",
    "madry-imagenet_linf_8": "xkcd:red",
    "geirhos-resnet50_trained_on_SIN": "xkcd:green",
    "geirhos-resnet50_trained_on_SIN_and_IN": "xkcd:green",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": "xkcd:green",
    "semi-supervised-YFCC100M": "xkcd:indigo",
    "semi-weakly-supervised-instagram": "xkcd:dark blue",
}

markers = {
    # "BERT_text": "BERT",
    # "GPT2_text": "GPT2",
    # "Word2Vec",
    "none": ("xkcd: black", "-"),
    "CLIP-RN50": ("xkcd:blue", "-"),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", "--"),
    "ICMLM": ("xkcd:blue", ":"),
    "TSM-v": ("xkcd:indigo", "-"),
    "TSM-vat": ("xkcd:indigo", "--"),
    "GPV": ("xkcd:light blue", "-"),
    "GPV-SCE": ("xkcd:light blue", "--"),
    "RN50": ("xkcd:orange", "-"),
    "BiT-M-R50x1": ("xkcd:puce", "-"),
    "madry-imagenet_l2_3_0": ("xkcd:red", "-"),
    "madry-imagenet_linf_4": ("xkcd:red", "--"),
    "madry-imagenet_linf_8": ("xkcd:red", ":"),
    "geirhos-resnet50_trained_on_SIN": ("xkcd:green", "-"),
    "geirhos-resnet50_trained_on_SIN_and_IN": ("xkcd:green", "--"),
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": ("xkcd:green", ":"),
    "semi-supervised-YFCC100M": ("xkcd:indigo", "-"),
    "semi-weakly-supervised-instagram": ("xkcd:indigo", "--"),
}

markers_bars = {
    # "BERT_text": "BERT",
    # "GPT2_text": "GPT2",
    # "Word2Vec",
    "none": ("xkcd:black", ""),
    "GPT2": ("xkcd:dark grey", ""),
    "BERT": ("xkcd:light grey", ""),
    "CLIP-RN50": ("xkcd:light blue", ""),
    # "semi-supervised-YFCC100M": ("xkcd:indigo", ""),
    # "semi-weakly-supervised-instagram": ("xkcd:dark blue", ""),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", ""),
    "ICMLM": ("xkcd:cyan", ""),
    "TSM-v": ("xkcd:bright blue", ""),
    "TSM-vat": ("xkcd:cornflower blue", ""),
    "GPV": ("xkcd:light blue", "/"),
    "GPV-SCE": ("xkcd:cyan", "/"),
    "RN50": ("xkcd:orange", ""),
    "BiT-M-R50x1": ("xkcd:puce", ""),
    "madry-imagenet_l2_3_0": ("xkcd:light red", ""),
    "madry-imagenet_linf_4": ("xkcd:red", ""),
    "madry-imagenet_linf_8": ("xkcd:dark red", ""),
    "geirhos-resnet50_trained_on_SIN": ("xkcd:light green", ""),
    "geirhos-resnet50_trained_on_SIN_and_IN": ("xkcd:green", ""),
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": ("xkcd:forest green", ""),
}

chance_levels = {
    "HouseNumbers": 1 / 10,
    "CUB": 1 / 200,
    "CIFAR100": 1 / 100,
    "MNIST": 1 / 10,
    "FashionMNIST": 1 / 10,
    "CIFAR10": 1 / 10,
    "Caltech101": 1 / 101,
    "Caltech256": 1 / 256,
    "DTD": 1 / 47,
    "FGVC-Aircraft": 1 / 102,
    "Food101": 1 / 101,
    "Flowers102": 1 / 102,
    "IIITPets": 1 / 37,
    "SUN397": 1 / 397,
    "StanfordCars": 1 / 196,
    "Birdsnap": 1 / 500
}


class PlotMarker:
    def __init__(self, n_markers=10):
        self.markers = ["o", "x", ".", "v", "^", "*", "D", "+"]
        self.colors = ["b", "g", "r", "c", "m", "y", "k"]

        self.possible_markers = [f"{self.markers[k % len(self.markers)]}-{self.colors[k % len(self.colors)]}"
                                 for k in range(n_markers)]

        self.marker_count = 0

    def reset(self):
        self.marker_count = 0

    def get_marker(self):
        marker = self.possible_markers[self.marker_count]
        self.marker_count = (self.marker_count + 1) % len(self.possible_markers)
        return marker


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


clip_paper_results = {
    "val_acc": {
        "CLIP-RN50": {
            "Birdsnap": [56.4],
            "Caltech101": [89.6],
            "DTD": [76.4],
            "FGVC-Aircraft": [49.1],
            "Flowers102": [96.1],
            "IIITPets": [88.2],
            "SUN397": [73.3],
            "StanfordCars": [78.3],
            "CIFAR10": [88.7],
            "CIFAR100": [70.3],
            "Food101": [86.4]
        },
        "BiT-M-R50x1": {
            "Birdsnap": [70.9],
            "Caltech101": [93.9],
            "DTD": [77.3],
            "FGVC-Aircraft": [55.6],
            "Flowers102": [99.4],
            "IIITPets": [91.5],
            "SUN397": [69.9],
            "StanfordCars": [59.0],
            "CIFAR10": [94.9],
            "CIFAR100": [82.2],
            "Food101": [83.3]
        },
        "RN50": {
            "Food101": [71.3],
            "CIFAR10": [91.8],
            "CIFAR100": [74.5],
            "Birdsnap": [52.7],
            "SUN397": [60.5],
            "StanfordCars": [49.9],
            "FGVC-Aircraft": [48.5],
            "VOC2007": [],
            "DTD": [72.3],
            "IIITPets": [92.4],
            "Caltech101": [90.8],
            "Flowers102": [90.8],
        },
        "semi-weakly-supervised-instagram": {
            "Food101": [84.8],
            "CIFAR10": [95.9],
            "CIFAR100": [80.9],
            "Birdsnap": [63.8],
            "SUN397": [69.0],
            "StanfordCars": [74.2],
            "FGVC-Aircraft": [56.0],
            "VOC2007": [],
            "DTD": [75.4],
            "IIITPets": [95.4],
            "Caltech101": [93.9],
            "Flowers102": [91.7],
        },
        "virtex": {
            "Food101": [57.9],
            "CIFAR10": [83.9],
            "CIFAR100": [57.5],
            "Birdsnap": [17.0],
            "SUN397": [49.8],
            "StanfordCars": [22.4],
            "FGVC-Aircraft": [34.5],
            "VOC2007": [],
            "DTD": [58.2],
            "IIITPets": [53.6],
            "Caltech101": [70.6],
            "Flowers102": [74.7],
        }
    }
}

size_training_data = {
    "CLIP-RN50": 400e6,
    "virtex": 120e3,
    "ICMLM": 120e3,
    "GPV": 120e-3,
    "GPV-SCE": 120e-3,  # TODO
    "TSM-v": 120e6 * 32,
    "TSM-visual": 120e6 * 32,
    "RN50": 1.3e6,
    "BiT-M-R50x1": 14e6,
    "madry-imagenet_l2_3_0": 110 * 1.3e6,
    "madry-imagenet_linf_4": 110 * 1.3e6,
    "madry-imagenet_linf_8": 110 * 1.3e6,
    "geirhos-resnet50_trained_on_SIN": 1.3e6,
    "geirhos-resnet50_trained_on_SIN_and_IN": 2 * 1.3e6,
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": 2 * 1.3e6,
}


@dataclass(frozen=True)
class PlotConfig:
    y_ticks_font_size = 16
    x_ticks_font_size = 16
    legend_font_size = 20
    title_font_size = 20
    y_label_font_size = 20
    x_label_font_size = 20

plot_config = PlotConfig()
