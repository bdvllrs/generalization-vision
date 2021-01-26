from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import load_results


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
    result_id = 7
    idx_prototypes_bar_plot = 1

    accuracies, confusion_matrices, config = load_results(Path(f"results/{result_id}"))

    plot_marker = PlotMarker(len(config['model_names']))

    n_datasets = len(accuracies[list(accuracies.keys())[0]].keys())
    n_cols = 3
    n_rows = round(n_datasets / n_cols)

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.4)

    for k, dataset in enumerate(config['datasets']):
        plot_marker.reset()
        ax = fig.add_subplot(gs[k // 3, k % 3])
        for model, model_accuracies in accuracies.items():
            if dataset['name'] in model_accuracies:
                items = sorted(model_accuracies[dataset['name']].items(), key=lambda x: x[0])
                x, y = zip(*items)
                y_mean, y_std = zip(*y)

                ax.errorbar(x, y_mean, y_std, None, plot_marker.get_marker(), label=model)

        if k == 0:
            ax.legend()
        ax.set_title(dataset['name'])
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Number of prototypes per class")
    fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"results/{result_id}/plot.svg", format="svg")
    plt.show()

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.4)

    for k, dataset in enumerate(config['datasets']):
        plot_marker.reset()
        ax = fig.add_subplot(gs[k // 3, k % 3])
        for i, (model, model_accuracies) in enumerate(accuracies.items()):
            if dataset['name'] in model_accuracies:
                items = sorted(model_accuracies[dataset['name']].items(), key=lambda x: x[0])
                x, y = zip(*items)
                mean, std = zip(*y)
                ax.bar([i * 0.35], mean[idx_prototypes_bar_plot], 0.35, yerr=std[idx_prototypes_bar_plot], label=model)

        if k == 0:
            ax.legend()
        ax.set_title(dataset['name'])
        ax.set_ylabel("Accuracy")
    fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"results/{result_id}/plot_bar.svg", format="svg")
    plt.show()

    print(config)
