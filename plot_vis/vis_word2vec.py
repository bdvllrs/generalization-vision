import gensim
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import LineSentence, Word2Vec

model_names = [
    "none",
    # "RN50",
    # "CLIP-RN50",
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

if __name__ == '__main__':
    loss_path = "../wordvectors/losses_{model_name}.npy"
    model_path = "../wordvectors/{model_name}.model"
    val_dataset = LineSentence("/mnt/SSD/datasets/enwiki/wiki.en.val.text")

    # for model_name in model_names:
    #     losses = np.load(loss_path.format(model_name=model_name))
    #     print("ok")

    for model_name in model_names:
        path = model_path.format(model_name=model_name)
        print("Loading", path)
        model = Word2Vec.load(path)
        print(model_name, "log prob", np.mean(model.score(val_dataset, 929628)))



