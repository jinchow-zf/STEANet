# STEANet: Efficiently learning spatiotemporal affective representations from videos

This repo is the official implementation of "STEANet: Efficiently learning spatiotemporal affective representations from videos".

## Introduction

In this work, we propose a novel Spatiotemporal Emotion Adaptation Network for recognizing emotions in a parameter efficient manner. The framework of the proposed method is shown as below.

<p><img src="figure/figure.jpg" width="800" /></p>


## Installation

The codes are based on [AIM](https://github.com/taoyang1122/adapt-image-models), which is based on [MMAction2](https://github.com/open-mmlab/mmaction2). Thanks for their awesome works! To prepare the environment, please follow the following instructions.
```shell
# create virtual environment
conda create -n STEANet python=3.7.13
conda activate STEANet

# install pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install other requirements
pip install -r requirements.txt

# install clip
pip install git+https://github.com/openai/CLIP.git

# install mmaction2
python setup.py develop
```
## Apex
We use apex for mixed precision training by default. Please refer to the official [installation](https://github.com/NVIDIA/apex).

## Datasets
The used datasets are provided in [VideoEmotion-8](https://drive.google.com/drive/folders/0B5peJ1MHnIWGd3pFbzMyTG5BSGs?resourcekey=0-hZ1jo5t1hIauRpYhYIvWYA) and [Ekman-6](https://github.com/kittenish/Frame-Transformer-Network). The train/test splits in both two datasets follow the official procedure. To prepare the data, you can refer to [MMAction2](https://github.com/open-mmlab/mmaction2) for a general guideline.

## Training
We use the CLIP checkpoints from the [official release](https://github.com/openai/CLIP). In our experiments, we choose ViT-B/16 as the image backbone by default, as it achieves a balance between
performance and computational complexity.

The training configurations for different experiments on the two datasets are provided in `configs/recognition/vit/`. To run the experiments, please use the following command. Replace `PATH/TO/CONFIG` with the path to the training configuration you want to use, and `PATH/TO/OUTPUT` with the directory where you want to save the output.
```shell
bash tools/dist_train.sh <PATH/TO/CONFIG> <NUM_GPU> --test-last --validate --cfg-options work_dir=<PATH/TO/OUTPUT>
```

## Evaluation
The code will do the evaluation after training. If you would like to evaluate a model only, please use the following command.
```shell
bash tools/dist_test.sh <PATH/TO/CONFIG> <CHECKPOINT_FILE> <NUM_GPU> --eval top_k_accuracy
```

## Models
We now provide the model weights for VideoEmotion-8 and Ekman-6 datasets in the following [link](https://pan.baidu.com/s/1X1ssWW4PGU3LJFwIw6LsjQ?pwd=GNNU).


Cheerfully, our work is under review. If you are interested in our work, please email to [jinchow21@sina.com](jinchow21@sina.com).


