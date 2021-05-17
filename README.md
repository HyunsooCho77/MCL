# MCL on (CIFAR-10, CIFAR-100) 
Code for MCL.
This code includes SEI eval on CIFAR-100. (AUROC around 94.)


## Dependencies
* pytorch >=1.2
* torchvision >=0.4.0
* hydra >=0.11.3
* tqdm >=4.45.0

### Install Hydra
[Hydra](https://hydra.cc/docs/next/intro/#installation) is a python framework to manage the hyperparameters during
 training and evaluation. Install with:
 
 ``pip install hydra-core --upgrade``


## Usage

Dataset download and preprocess (4-way rotations augmented dataset with rotation label)
``python utils/data_preprocess.py``

Train MCL:
``python mcl_main.py``

Use the following prefix to train MCL with single GPU :
``CUDA_VISIBLE_DEVICES="GPU_number"``


All the hyperparameters are available in ``mcl_config.yml``, 
which could be overrided from the command line.

## Pretrained model.

MCL pretrained model.

https://www.dropbox.com/s/hwag0bp6e6cbmab/epoch_800.pt?dl=0

place pretrained model (``epoch_800.pt``) in  following directory

``./logs/model_name(default : MCL)/ckpt/epoch_800.pt``

to run pretrained model, set resume flage to True and load_epoch to 800 in ``mcl_config.yml``.

Then run ``python mcl_main.py``.

Experimental results are stored in ``./logs/model_name(default : MCL)/SEI performance_summary.txt`` file.

## Notes

Current version of SEI is not optimized enough, thus it might be slow.