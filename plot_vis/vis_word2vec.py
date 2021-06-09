import gensim
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence, Word2Vec

model_names = [
    "none",
    "RN50",
    "CLIP-RN50",
    # "BiT-M-R50x1",
    # "madry-imagenet_l2_3_0",
    # "virtex",
    # "geirhos-resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
    # "TSM-v",
    # "ICMLM",
    # "GPT2",
    # "BERT",
    # "madry-imagenet_linf_8",
    # "geirhos-resnet50_trained_on_SIN_and_IN",
    # "madry-imagenet_linf_4",
    # "geirhos-resnet50_trained_on_SIN",
]

scores = {
    "none": -25857.68
}

ntokens_train = 2227749224
ntokens_val = 372710618

if __name__ == '__main__':
    loss_path = "../wordvectors/300d/losses_{model_name}.npy"
    model_path = "../wordvectors/300d/{model_name}.model"
    analogiy_path = "/home/romain/W2V_evaluation"
    val_dataset = LineSentence("/mnt/SSD/datasets/enwiki/wiki.en.val.text")

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

    for model_name in model_names:
        model = gensim.models.Word2Vec.load(model_path.format(model_name=model_name))

        analogies = model.wv.evaluate_word_analogies(f'{analogiy_path}/questions-words.txt')
        print(model_name, ": ", analogies[0])

        analogies = model.wv.evaluate_word_pairs(f'{analogiy_path}/wordsim353.tsv')
        print(model_name, ": ", analogies[0])

    # for model_name in model_names:
    #     path = model_path.format(model_name=model_name)
    #     print("Loading", path)
    #     model = Word2Vec.load(path)
    #     print(model_name, "log prob", np.mean(model.score(val_dataset, 929628)))



