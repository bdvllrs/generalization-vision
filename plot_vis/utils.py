model_names_short = {
    "BERT_text": "BERT",
    "GPT2_text": "GPT2",
    # "Word2Vec",
    "CLIP-RN50": "CLIP",
    "CLIP-RN50_text": "CLIP-T",
    "RN50": "RN50",
    "virtex": "VirTex",
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
    "CLIP-RN50": ("xkcd:blue", "-"),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", "--"),
    "RN50": ("xkcd:orange", "-"),
    "BiT-M-R50x1": ("xkcd:puce", "-"),
    "madry-imagenet_l2_3_0": ("xkcd:red", "-"),
    "madry-imagenet_linf_4": ("xkcd:red", "--"),
    "madry-imagenet_linf_8": ("xkcd:red", ":"),
    "geirhos-resnet50_trained_on_SIN": ("xkcd:green", "-"),
    "geirhos-resnet50_trained_on_SIN_and_IN": ("xkcd:green", "--"),
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": ("xkcd:green", ":"),
}

markers_bars = {
    # "BERT_text": "BERT",
    # "GPT2_text": "GPT2",
    # "Word2Vec",
    "CLIP-RN50": ("xkcd:light blue", ""),
    "semi-supervised-YFCC100M": ("xkcd:indigo", ""),
    "semi-weakly-supervised-instagram": ("xkcd:dark blue", ""),
    # "CLIP-RN50_text": ("xkcd:indigo", "."),
    "virtex": ("xkcd:blue", ""),
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
