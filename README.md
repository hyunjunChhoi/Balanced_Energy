
Our code heavily relies on the implementation of [Energy-based Out-of-distribution Detection](https://github.com/wetliu/energy_ood) 
and [PEBAL](https://github.com/tianyu0207/PEBAL)

## Prerequisite

### Prepare Dataset 

#### Segmentation
We follow the installation process of [PEBAL](https://github.com/tianyu0207/PEBAL/blob/main/docs/installation.md)

All data tree has to be inserted in path:  Balanced_Energy/segmentation/code/dataset

#### Classification

We use cifar10, cifar100 as training data

We use auxiliary data as 300K random images following [Outlier Exposure](https://github.com/hendrycks/outlier-exposure)

We test on the SC-OOD benchmark ,this should be inserted in data tree 
which can be downloaded from [SC-OOD UDG](https://github.com/Jingkang50/ICCV21_SCOOD)

```shell
classification/data
├── cifar10
├── cifar100
├── data 
│       ├── images
│       └── imglist
└── tinyimages80m
    └── 300K_random_images.npy

```


### Install dependencies

The project is based on the pytorch 1.8.1 with python 3.8.

1) create conda env
    ```shell
    $ conda env create -f balanced.yml
    ```
2) install the torch 1.8.1
    ```shell
    $ conda activate balanced
    # IF cuda version < 11.0
    $ pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    # IF cuda version >= 11.0 (e.g., 30x or above)
    $ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

### Prepare checkpoint

All checkpoint is in the zip files following google drive link:

[checkpoint](https://drive.google.com/file/d/1V9STZyI4uQ1x_eckkdryfYj36QZzCBnE/view?usp=share_link)


#### Segmentation

put the checkpoint in the path : Balanced_Energy/segmentation/ckpts/pebal_balanced/

additionally, to finetune from the nvidia cityscapes model,

we follow [Meta-OoD](https://github.com/robin-chan/meta-ood) and use the deeplabv3+ checkpoint
in [here](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet). you'll need to put it in "ckpts/pretrained_ckpts" directory, and
**please note that downloading the checkpoint before running the code is necessary**


#### Classification

put the checkpoint in the path : Balanced_Energy/classification/snapshots/pretrained

## Segmentation code run

#### Data path Setting in Config File

Open 
Balanced_Energy/segmentation/code/config/config.py

Set the root path for datasets


#### Use package run the training : main.py  

1) python code/main.py  in the  Balanced_Energy/segmentation/

#### Use package run the evaluation : test.py

2) python code/test.py  in the Balanced_Energy/segmentation/


## Classification code run


in the  Balanced_Energy/classification/

run ResNet18 balanced_energy_fine_tune training and testing for cifar10 with trial index  3
```train
bash inf_run_res.sh energy_ft 0 3 
```

run ResNet18  balanced_energy_fine_tune training and testing for cifar100 with trial index  3
```train
bash inf_run_res.sh energy_ft 1 3
```

run ResNet18 balanced_energy_fine_tune training and testing for cifar10-LT with trial index  3
```train
bash inf_run_im_res.sh energy_ft 0 3 
```

run ResNet18  balanced_energy_fine_tune training and testing for cifar100-LT with trial index  3
```train
bash inf_run_im_res.sh energy_ft 1 3
```

the setting of hyperparameter alpha and gamma can be controlled in the bash script

