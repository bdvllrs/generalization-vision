# Few Shot
Few Shot comparison of models

## Quickstart
Download MadryLab's robust pre-trained models for ImageNet here: https://github.com/MadryLab/robustness#pretrained-models
and change `madry_model_folder` in `utils.py`.

The default is `$TORCH_HOME/checkpoints` and the default of `$TORCH_HOME` is `~/.cache/torch`.

### Compute accuracies
Execute `main.py`

### Analyze results
Execute `analyze.py` and choose corresponding `result_id`.
