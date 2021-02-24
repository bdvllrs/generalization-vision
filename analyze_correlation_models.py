import json
from pathlib import Path
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram

model_names_short = {
    "BERT_text": "BERT",
    "GPT2_text": "GPT2",
    # "Word2Vec",
    "CLIP-RN50": "CLIP",
    "CLIP-RN50_text": "CLIP-T",
    "RN50": "RN50",
    "virtex": "VIRT",
    "BiT-M-R50x1": "BiT-M",
    "madry-imagenet_l2_3_0": "MAD-L2",
    "madry-imagenet_linf_4": "MAD-LI4",
    "madry-imagenet_linf_8": "MAD-LI8",
    "geirhos-resnet50_trained_on_SIN": "GEI",
    "geirhos-resnet50_trained_on_SIN_and_IN": "GEI-IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN": "GEI-FIN",
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


def load_corr_results(results_path):
    with open(results_path / "config.json", "r") as f:
        config = json.load(f)

    corr = np.load(results_path / "correlations.npy", allow_pickle=True).item()
    feature_cache = np.load(results_path / "feature_cache.npy", allow_pickle=True).item()
    return corr, feature_cache, config

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == '__main__':
    result_id = 40
    idx_prototypes_bar_plot = 1
    dataset = "ImageNet"

    correlations, features, config = load_corr_results(Path(f"results/{result_id}"))

    y, X = zip(*features.items())
    y_short = [model_names_short[name] for name in y]
    X = np.stack(X, axis=0)

    # tSNE
    tsne = TSNE(n_components=2, perplexity=3, metric="correlation")
    X_pca = tsne.fit_transform(X)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for xc, yc, t in zip(X_pca[:, 0], X_pca[:, 1], y_short):
        plt.text(xc, yc, t)
    plt.title("t-SNE of RDMs")
    plt.savefig(f"results/{result_id}/tsne_rdms.jpg", format="jpg")
    plt.show()

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for xc, yc, t in zip(X_pca[:, 0], X_pca[:, 1], y_short):
        plt.text(xc, yc, t)
    plt.title("PCA of RDMs")
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.savefig(f"results/{result_id}/pca_rdms.jpg", format="jpg")
    plt.show()

    # Dendogram
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="correlation", linkage="average")
    model = model.fit(X)
    plt.figure(figsize=(10, 10))
    plt.title("Dendogram of hierarchical clustering of RDMs. Average linkage.")
    plot_dendrogram(model, labels=y_short)
    plt.savefig(f"results/{result_id}/dendogram_hierarchical_clustering_rdms.jpg", format="jpg")
    plt.show()

    mat = np.zeros((len(correlations), len(correlations)))
    labels = []

    # Correlation between models
    for i, (model_1, corrs) in enumerate(sorted(correlations.items())):
        labels.append(model_names_short[model_1])
        for j, (model_2, corr) in enumerate(sorted(corrs.items())):
            mat[i, j] = corr

    plt.figure(figsize=(10, 10))
    sn.heatmap(mat, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Pearson correlations between RDMs of vision and text models.")
    plt.savefig(f"results/{result_id}/plot_corr.svg", format="svg")
    plt.savefig(f"results/{result_id}/corr_rdms.jpg", format="jpg")
    plt.show()

