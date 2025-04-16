Entry points for either pretraining the contrastive model, training a DANN, or training a baseline classifier is:
```
$ python train.py --help 
usage: train.py [-h] [--print_iter PRINT_ITER] config

positional arguments:
  config

options:
  -h, --help            show this help message and exit
  --print_iter PRINT_ITER
                        zero for never, -1 for every epoch
```
Entry point for finetuning a pretrained contrastive model is:
```
$ python fine_tune_clr.py --help 
usage: fine_tune_clr.py [-h] [--pickle_model] [--pickle_name PICKLE_NAME] [--no_augs] [--max_iter MAX_ITER] [--nominal_xtalk NOMINAL_XTALK] config finetune_data_path clr_weights

positional arguments:
  config
  finetune_data_path
  clr_weights

options:
  -h, --help            show this help message and exit
  --pickle_model
  --pickle_name PICKLE_NAME
  --no_augs
  --max_iter MAX_ITER
  --nominal_xtalk NOMINAL_XTALK
```
Entry point for evaluating a model on the test dataset is:
```
$ python make_preds.py --help 
usage: make_preds.py [-h] [--classifier | --clr | --dann] [--finetune_pickle FINETUNE_PICKLE] [--xtalk XTALK] [--batch_mode] [--small_output] config weights test_data_path preds_fname

positional arguments:
  config
  weights
  test_data_path
  preds_fname

options:
  -h, --help            show this help message and exit
  --classifier
  --clr
  --dann
  --finetune_pickle FINETUNE_PICKLE
  --xtalk XTALK
  --batch_mode
  --small_output

```

Experiments are configured via yaml files in `experiments/`, see `config_parser.py` for valid and required fields. Saved models + losses appear in a study in `checkpoints/` in a directory named after the experiment.

Datasets are expected in the format:
```
train/
  electron/
  muon/
  pion/
  proton/
  # Any other classes
val/
  ^^
test/
  ^^
```
See `dataset.py` for the expected format.

The script `../larnd/convert_data.py` was used to convert larnd-sim files into the desired format.

For the segmented scintillator cube dataset [link](https://zenodo.org/records/10998285), the original file format is used. There was an issue with many of the events in a particle class sharing the same true kinematics which was fixed by ensuring only events within each of train/val/test have a unique set of true kinematic variables. This was done by running `misc/summarise_dataset.py` and then `misc/make_truly_random_segcube_dataset.py` which the hardcoded variables at the top of each script pointing to the relevant dataset locations.
