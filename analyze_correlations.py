from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import load_results, load_corr_results


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


if __name__ == '__main__':
    result_id = 228
    idx_prototypes_bar_plot = 1
    dataset = "ImageNet"

    rsa_corr_bert, rsa_corr_resnet, resnet_bert_score, config = load_corr_results(Path(f"results/{result_id}"))


    fig = plt.figure(figsize=(5, 5))
    plt.bar([0], [1], 0.35, label="BERT")
    for k, (model_name, model) in enumerate(resnet_bert_score.items(), 1):
        mean, std = resnet_bert_score[model_name][dataset][-1]
        plt.bar([k * 0.35], [mean], 0.35, label=model_name)
    plt.legend()
    plt.title("ResNet-BERT score")
    plt.savefig(f"results/{result_id}/plot_corr.svg", format="svg")
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    for k, (model_name, model) in enumerate(rsa_corr_bert.items(), 1):
        mean, std = rsa_corr_bert[model_name][dataset][-1]
        plt.bar([k * 0.35], [mean], 0.35, label=model_name)
    plt.legend()
    plt.title("BERT correlations")
    plt.savefig(f"results/{result_id}/plot_corr_bert.svg", format="svg")
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    for k, (model_name, model) in enumerate(rsa_corr_resnet.items(), 1):
        mean, std = rsa_corr_resnet[model_name][dataset][-1]
        plt.bar([k * 0.35], [mean], 0.35, label=model_name)
    plt.legend()
    plt.title("ResNet50 correlations")
    plt.savefig(f"results/{result_id}/plot_corr_resnet.svg", format="svg")
    plt.show()
