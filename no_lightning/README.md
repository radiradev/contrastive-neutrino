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

Experiments are configured via yaml files in `experiments/`, see `config_parser.py` for valid and required fields. Saved models + losses appear in a study in `checkpoints/` in a directory names after the experiment.
