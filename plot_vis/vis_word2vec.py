import argparse
import gensim
import os

import numpy as np
import scipy
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
from matplotlib import pyplot as plt

from plot_vis.utils import model_names_short, markers_bars, plot_config
from visiongeneralization.utils import load_conf, get_bootstrap_estimates

model_order = list(reversed([
    "BiT-M-R50x1",
    "geirhos-resnet50_trained_on_SIN",
    "geirhos-resnet50_trained_on_SIN_and_IN",
    "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    "RN50",
    "madry-imagenet_l2_3_0",
    "madry-imagenet_linf_4",
    "madry-imagenet_linf_8",
    "CLIP-RN50",
    "virtex",
    "TSM-v",
    "ICMLM",
    "none"
    # "TSM-vat",
    # "semi-supervised-YFCC100M", "semi-weakly-supervised-instagram"
]))

val_loss = {
    "none": 0.19160065893266287,
    "RN50": 0.19087196490871103,
    "CLIP-RN50": 0.1911929216891803,
    "BERT": 0.1926843093050813,
    "virtex": 0.18982931149012772,
    "GPT2": 0.19209159208874485
}

ntokens_train = 2227749224
ntokens_val = 372710618

def get_per_sample_acc(confusion_matrix):
    acc = []
    for i in range(confusion_matrix.shape[0]):
        for j in range(i, confusion_matrix.shape[1]):
            if i == j:
                acc.extend([1] * confusion_matrix[i, j])
            elif confusion_matrix[i, j] > 0:
                acc.extend([0] * confusion_matrix[i, j])
    return np.array(acc)


if __name__ == '__main__':
    conf = load_conf()
    parser = argparse.ArgumentParser(description='Unsupervised clustering visualisations')
    parser.add_argument('--word_vectors', default="../wordvectors", type=str,
                        help='Path to word vectors.')
    args = parser.parse_args()

    loss_path = os.path.join(args.word_vectors, "300d/losses_{model_name}.npy")
    model_path = os.path.join(args.word_vectors, "300d/{model_name}_epoch-4.model")
    analogiy_path = conf.analogy_path
    # val_dataset = LineSentence("/mnt/SSD/datasets/enwiki/wiki.en.val.text")

    # for model_name in model_names:
    #     cum_losses = np.load(loss_path.format(model_name=model_name))
    #     losses = []
    #     for k in range(len(cum_losses)):
    #         if k == 0:
    #             losses.append(cum_losses[k])
    #         else:
    #             losses.append(cum_losses[k] - cum_losses[k-1])
    #     losses = np.array(losses)
    #     print(model_name)
    #     print(cum_losses)
    #     print(losses / ntokens_train)
    sanity_check_accuracies = np.load("../results/cls_perf_sanity_checks_emb_dim_300.npy", allow_pickle=True).item()

    analogies = {}
    analogies_morphology = {}
    word_pairs = {}
    acc_vec = []
    analogy_vec = []
    analogy_morphology_vec = []
    word_pair_vec = []

    for model_name in model_order:
        model = gensim.models.Word2Vec.load(model_path.format(model_name=model_name))

        analogies[model_name] = model.wv.evaluate_word_analogies(f'{analogiy_path}/question-words-without-morphology.txt')
        analogies_morphology[model_name] = model.wv.evaluate_word_analogies(f'{analogiy_path}/question-words-only-morphology.txt')
        word_pairs[model_name] = model.wv.evaluate_word_pairs(f'{analogiy_path}/wordsim353.tsv')

        if model_name in sanity_check_accuracies['accuracies']:
            acc_per_sample = get_per_sample_acc(sanity_check_accuracies['confusion'][model_name])
            mean_bootstrap, std_bootstrap, interval = get_bootstrap_estimates(acc_per_sample, 50_000)
            print(f"{model_name} significant: {mean_bootstrap > 1/824 and mean_bootstrap - 1/824 > interval}")
            acc_vec.append(sanity_check_accuracies['accuracies'][model_name])
            analogy_vec.append(analogies[model_name][0])
            analogy_morphology_vec.append(analogies_morphology[model_name][0])
            word_pair_vec.append(word_pairs[model_name][0][0])
    print(f"Correlation to analogy: {np.corrcoef(np.array(acc_vec), np.array(analogy_vec))}")
    print(f"Correlation to analogy (morphology): {np.corrcoef(np.array(acc_vec), np.array(analogy_morphology_vec))}")
    print(f"Correlation to word_pair: {np.corrcoef(np.array(acc_vec), np.array(word_pair_vec))}")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(27, 9))

    for k, model_name in enumerate(model_order):
        if model_name in model_names_short:
            color, hatch = markers_bars[model_name]
            analogy, err = analogies[model_name]
            analogy_m, err_m = analogies_morphology[model_name]
            word_pair, spearman, _ = word_pairs[model_name]
            axes[0].bar([k * 0.35], analogy, 0.35, color=color, hatch=hatch,
                        label=model_names_short[model_name])
            axes[1].bar([k * 0.35], analogy_m, 0.35, color=color, hatch=hatch)
            axes[2].bar([k * 0.35], word_pair[0], 0.35, yerr=word_pair[1], color=color, hatch=hatch)

    axes[0].set_title(f"Word Analogy (semantic)")
    axes[0].set_ylim(0.5, 0.8)
    axes[1].set_title(f"Word Analogy (morphology)")
    axes[1].set_ylim(0.5, 0.8)
    axes[2].set_ylim(0.3, 0.7)
    axes[2].set_title(f"Word Pair similarity")
    axes[0].set_ylabel("Accuracy")
    for k in range(3):
        axes[k].set_xticks([])
        axes[k].set_xlabel("")

        axes[k].yaxis.label.set_size(2 * plot_config.y_label_font_size)
        axes[k].xaxis.label.set_size(2 * plot_config.x_label_font_size)
        axes[k].title.set_size(2 * plot_config.title_font_size)
        axes[k].tick_params(axis='y', labelsize=2 * plot_config.y_ticks_font_size)
        axes[k].tick_params(axis='x', labelsize=2 * plot_config.x_ticks_font_size)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=3, fontsize=2 * plot_config.legend_font_size)
    # fig.legend(fontsize=2 * plot_config.legend_font_size)
    # fig.legend()
    plt.tight_layout(pad=.5)
    # fig.suptitle("Few-shot accuracies on various datasets and models")
    plt.savefig(f"../results/linguistic_generalization.svg", format="svg")
    plt.show()

    # for model_name in model_names:
    #     path = model_path.format(model_name=model_name)
    #     print("Loading", path)
    #     model = Word2Vec.load(path)
    #     print(model_name, "log prob", np.mean(model.score(val_dataset, 929628)))
