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
import umap

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
    significance = dict()
    if (results_path / "correlations.npy").exists():
        significance = np.load(results_path / "correlations.npy", allow_pickle=True).item()
    feature_cache = np.load(results_path / "feature_cache.npy", allow_pickle=True).item()
    dim_reducted_features = {}
    if (results_path / "dim_reducted_features.npy").exists():
        dim_reducted_features = np.load(results_path / "dim_reducted_features.npy", allow_pickle=True).item()
    return corr, significance, feature_cache, dim_reducted_features, config

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
    result_id = 230
    idx_prototypes_bar_plot = 1

    dataset = "ImageNet"

    figsize = 5

    correlations, significance, features, dim_reduced_features, config = load_corr_results(Path(f"results/{result_id}"))


    # # UMAP
    # reducer = umap.UMAP(n_components=2, min_dist=0.05, n_neighbors=5, metric="correlation")
    # x_umap = reducer.fit_transform(X)
    # x_umap = dim_reduced_features['umap']
    # y_short = [model_names_short[name] for name in dim_reduced_features['labels']]
    # colors = [color_scheme[name] for name in dim_reduced_features['labels']]
    # plt.figure(figsize=(figsize, figsize))
    # plt.scatter(x_umap[:, 0], x_umap[:, 1], c=colors)
    # for xc, yc, t in zip(x_umap[:, 0], x_umap[:, 1], y_short):
    #     plt.text(xc, yc, t)
    # # plt.title("UMAP of RDMs")
    # plt.tight_layout(.5)
    # plt.savefig(f"results/{result_id}/umap_rdms.eps", format="eps")
    # plt.show()

    # # tSNE
    # tsne = TSNE(n_components=2, perplexity=3, learning_rate=1, min_grad_norm=0, metric="correlation")
    # X_tsne = tsne.fit_transform(X)
    X_tsne = dim_reduced_features['tsne']
    y_short = [model_names_short[name] for name in dim_reduced_features['labels']]
    colors = [color_scheme[name] for name in dim_reduced_features['labels']]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(1.5*figsize, 0.8*figsize))
    # ax = fig.add_subplot(111, projection='3d')
    ax = axes[1]
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)
    for xc, yc, t in zip(X_tsne[:, 0], X_tsne[:, 1], y_short):
        ax.text(xc, yc, t, ha='center')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))
    ax.set_title("(b)")
    # plt.title("t-SNE of RDMs")


    y, X = zip(*features.items())
    y_short = [model_names_short[name] for name in y]
    colors = [color_scheme[name] for name in y]
    X = np.stack(X, axis=0)

    # Dendrogram
    ax = axes[0]
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="correlation", linkage="average")
    model = model.fit(X)
    # plt.title("Dendrogram of hierarchical clustering of RDMs. Average linkage.")
    plot_dendrogram(model, labels=y_short, leaf_rotation="vertical", ax=ax)
    ax.set_title("(a)")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))
    plt.tight_layout(pad=1)
    plt.savefig(f"results/{result_id}/tsne_dendrogram_hierarchical_clustering_rdms_3.svg", format="svg")
    plt.show()

    # # PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    # plt.figure(figsize=(figsize, figsize))
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
    # for xc, yc, t in zip(X_pca[:, 0], X_pca[:, 1], y_short):
    #     plt.text(xc, yc, t)
    # # plt.title("PCA of RDMs")
    # # plt.xlabel("1st component")
    # # plt.ylabel("2nd component")
    # plt.tight_layout(.5)
    # plt.savefig(f"results/{result_id}/pca_rdms.eps", format="eps")
    # plt.show()
    #

    figsize = 10

    mat = np.zeros((len(correlations), len(correlations)))
    pval = np.zeros((len(significance), len(significance)))
    labels = []

    # Correlation between models
    for i, (model_1, corrs) in enumerate(sorted(correlations.items())):
        labels.append(model_names_short[model_1])
        for j, (model_2, corr) in enumerate(sorted(corrs.items())):
            mat[i, j] = corr
            pval = significance[model_1][model_2]

    plt.figure(figsize=(figsize, figsize))
    sn.heatmap(mat, annot=True, xticklabels=labels, yticklabels=labels)
    # plt.title("Pearson correlations between RDMs of vision and text models.")
    plt.tight_layout(pad=.5)
    plt.savefig(f"results/{result_id}/plot_corr.eps", format="eps")
    plt.show()

    # plt.figure(figsize=(figsize, figsize))
    # sn.heatmap(pval, annot=True, xticklabels=labels, yticklabels=labels)
    # # plt.title("Pearson correlations between RDMs of vision and text models.")
    # plt.tight_layout(.5)
    # plt.savefig(f"results/{result_id}/plot_pval.eps", format="eps")
    # plt.show()

