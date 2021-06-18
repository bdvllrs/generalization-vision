import argparse
import gensim
import os
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence
from matplotlib import pyplot as plt

from plot_vis.utils import model_names_short, markers_bars, plot_config
from visiongeneralization.utils import load_conf

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

    analogies = {}
    word_pairs = {}

    for model_name in model_order:
        model = gensim.models.Word2Vec.load(model_path.format(model_name=model_name))

        analogies[model_name] = model.wv.evaluate_word_analogies(f'{analogiy_path}/questions-words.txt')
        word_pairs[model_name] = model.wv.evaluate_word_pairs(f'{analogiy_path}/wordsim353.tsv')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    for k, model_name in enumerate(model_order):
        if model_name in model_names_short:
            color, hatch = markers_bars[model_name]
            analogy, err = analogies[model_name]
            word_pair, spearman, _ = word_pairs[model_name]
            axes[0].bar([k * 0.35], analogy, 0.35, color=color, hatch=hatch,
                        label=model_names_short[model_name])
            axes[1].bar([k * 0.35], word_pair[0], 0.35, yerr=word_pair[1], color=color, hatch=hatch)

    axes[0].set_title(f"Word Analogy")
    axes[0].set_ylim(0.6, 0.75)
    axes[1].set_ylim(0.3, 0.7)
    axes[1].set_title(f"Word Pair similarity")
    axes[0].set_ylabel("Accuracy")
    for k in range(2):
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
