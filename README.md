# Does Language Help Generalization in Vision Models?
Code and configurations related to the paper:

Devillers, B., Choksi, B., Bielawski, R., & VanRullen, R. (2021). Does language help generalization in vision models? CoNLL 2021: Computational Natural Language Learning. [arXiv:2104.08313](https://arxiv.org/abs/2104.08313).

## Setup
Download and install the project:
```
git clone https://github.com/bdvllrs/generalization-vision.git
cd generalization-vision
pip install --user -e .
```

The code has been tested with python 3.8.5.

### Pretrained models
All pretrained models should be placed in `$TORCH_HOME/checkpoints`.

The default for `$TORCH_HOME` is `~/.cache/torch`. You can set the `$TORCH_HOME` environment
variable to update it.

### Adversarially Robust models
Download MadryLab's robust pre-trained models for ImageNet here: 
```
https://github.com/MadryLab/robustness#pretrained-models
```

Here we use the 3 ImageNet trained models:
- https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0
- https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0
- https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0

### BiT models
Download the BiT models here:
```
https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
```

### Configuration
The config file is located in `config/main.yaml`. 
Update the path of the datasets.


## Reproduce our results
The folder `scripts` contains the scripts we used to generate the results in the paper.

Running the script will only generate the data. To produce the plots, use the script `vis_{script_name}.py --load_result {checkpoint_id}`
where the `checkpoint_id` is the id of the experiment (in the `results` folder).

For the summary plot (figure 5 in the paper), use:
```
python vis_summary_plot.py --load_results_few_shot {ID_FEW_SHOT} --load_results_clustering {ID_CLUSTERING} --load_results_transfer_learning {ID_TRANSFER_LEARNING}
```

**NB. If you run the word2vec script with an enwiki dataset and want to try with another one, delete the file `tasks/ntokens.json` before re-running.**

## Pretrained models
The word2vec models (gensim models) trained using the method detailed in the paper are available in this repository: 
[https://github.com/bdvllrs/generalization-vision-pretrained](https://github.com/bdvllrs/generalization-vision-pretrained).

## API
Start the different generalization tasks from the `tasks` folder.
The tasks will create folders in the `results` folder with the id of the experiment.
It will contain the results and checkpoints.

### Few-shot learning task
```
python few_shot.py
```
#### Parameters:
- `--models` (`list[str]`) list of the models to use. Defaults to all models.
- `--ntrials` (`int`) number of repetitions per experiment. Defaults to 10.
- `--batch_size` (`int`) batch size. Defaults to 64.
- `--load_results` (`int`) id of a previous experiment to load and continue.
- `--override_models` (`list[str]`) if a previous experiment is loaded, list of models that had already been computed to recompute. Defaults to none. 

### Unsupervised clustering task
```
python unsupervised_clustering.py
```
#### Parameters:
- `--models` (`list[str]`) list of the models to use. Defaults to all models.
- `--batch_size` (`int`) batch size. Defaults to 64.
- `--load_results` (`int`) id of a previous experiment to load and continue.
- `--override_models` (`list[str]`) if a previous experiment is loaded, list of models that had already been computed to recompute. Defaults to none. 

### Transfer learning task
```
python transfer_learning.py
```
#### Parameters:
- `--models` (`list[str]`) list of the models to use. Defaults to all models.
- `--lr` (`float`) learning rate. Defaults to 1e-3.
- `--n_workers` (`int`) number of workers to use. Defaults to 0.
- `--n_epochs` (`int`) number of epochs to compute. Defaults to 20.  
- `--batch_size` (`int`) batch size. Defaults to 64.
- `--load_results` (`int`) id of a previous experiment to load and continue.
- `--override_models` (`list[str]`) if a previous experiment is loaded, list of models that had already been computed to recompute. Defaults to none. 

### Model Comparison
```
python correlations_models.py
```
#### Parameters:
- `--models` (`list[str]`) list of the models to use. Defaults to all models.
- `--batch_size` (`int`) batch size. Defaults to 64.
- `--rdm_distance_metric` (`str`) among `t-test`, `cosine`, `correlation`. Distance metric to use to compute the RDMs. Defaults to `t-test`.  
- `--rda_correlation_type` (`str`) among `pearson`, `spearman` and `kendall`. Correlation function to use to correlate the RDMs. Defaults to `spearson`.  
- `--load_results` (`int`) id of a previous experiment to load and continue.
- `--override_models` (`list[str]`) if a previous experiment is loaded, list of models that had already been computed to recompute. Defaults to none. 

### Skip-Grams
```
python word2vec.py
```
#### Parameters:
- `--models` (`list[str]`) list of the models to use. Defaults to all models.
- `--batch_size` (`int`) batch size. Defaults to 64.
- `--nepochs` (`int`) number of epochs to train. Defaults to 5.  
- `--vocab_size` (`int`) size of the vocabulary to use, not counting the visual words. Defaults to 20,000.
- `--emb_dimension` (`int`) dimension of the embeddings vectors. Visual word embeddings will be reduced using a PCA. If -1, the embedding size will be determined by the embedding size of the visual word embeddings. Defaults to 300.  
- `--window_size` (`int`)  Gensim's `window_size` parameter. Defaults to 5.
- `--save_dir` path to save the word embeddings to.
- `--load_dir` path to load the word embeddings from in the case of evaluating the word embeddings.
- `--load_results` (`int`) id of a previous experiment to load and continue.
- `--override_models` (`list[str]`) if a previous experiment is loaded, list of models that had already been computed to recompute. Defaults to none. 

#### Word Analogies & Word Similarities Datasets
The dataset to evaluate the Skip-Gram models are available in the gensim repository at https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_data/wordsim353.tsv for word similarities and https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_data/questions-words.txt for word analogies
