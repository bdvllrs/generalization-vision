import matplotlib.pyplot as plt
import numpy as np


class PlotMarker:
    def __init__(self, n_markers=10):
        self.markers = ["o", "x", ".", "v", "^", "*", "D", "+"]
        self.colors = ["b", "g", "r", "c", "m", "y", "k"]

        self.possible_markers = [f"{self.markers[k % len(self.markers)]}-{self.colors[k % len(self.colors)]}"
                                 for k in range(n_markers)]

        self.marker_count = 0

    def get_marker(self):
        marker = self.possible_markers[self.marker_count]
        self.marker_count = (self.marker_count + 1) % len(self.possible_markers)
        return marker


if __name__ == '__main__':
    result_id = 1

    config = np.load(f"results/{result_id}/config.npy", allow_pickle=True).item()
    accuracies = np.load(f"results/{result_id}/accuracies.npy", allow_pickle=True).item()
    confusion_matrices = np.load(f"results/{result_id}/confusion_matrices.npy", allow_pickle=True).item()

    plot_marker = PlotMarker(len(config['model_names']))

    for dataset in config['datasets']:
        fig, ax = plt.subplots(figsize=(10, 10))
        for model, model_accuracies in accuracies.items():
            x = list(model_accuracies[dataset['name']].keys())
            y = list(model_accuracies[dataset['name']].values())

            ax.plot(x, y, plot_marker.get_marker(), label=model)

        ax.legend()
        ax.set_title(f"Few-shot accuracies on {dataset['name']}.")
        plt.show()

    print(config)
