# Efficient Affective Representation Learning in User-Generated Videos via Spatiotemporal Emotion Adaptation Network

This repo is the official implementation of "Efficient Affective Representation Learning in User-Generated Videos via Spatiotemporal Emotion Adaptation Network".

## Introduction

In this work, we propose a novel Spatiotemporal Emotion Adaptation Network for recognizing emotions in a parameter efficient manner. The framework of the proposed method is shown as below.

<p><img src="figure/pipeline.jpg" width="800" /></p>


## Installation

The codes are based on [AIM][https://github.com/open-mmlab/mmaction2](https://github.com/taoyang1122/adapt-image-models), which is based on [MMAction2](https://github.com/open-mmlab/mmaction2). Thanks for their awesome works! To prepare the environment, please follow the following instructions.
```shell
# create virtual environment
conda create -n STEANet python=3.7.13
conda activate STEANet

# install pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# install CLIP
pip install git+https://github.com/openai/CLIP.git

# install other requirements
pip install -r requirements.txt

# install mmaction2
python setup.py develop
```
### Install Apex:
We use apex for mixed precision training by default. To install apex, please follow the instructions in the [repo](https://github.com/NVIDIA/apex).

If you would like to disable apex, comment out the following code block in the [configuration files](configs/recognition/vit/):
```
# do not use mmcv version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Training
The training configs of different experiments are provided in `configs/recognition/vit/`. To run experiments, please use the following command. `PATH/TO/CONFIG` is the training config you want to use. The default training setting is 8GPU with a batchsize of 64.
```shell
bash tools/dist_train.sh <PATH/TO/CONFIG> <NUM_GPU> --test-last --validate --cfg-options work_dir=<PATH/TO/OUTPUT>
```
We also provide a training script in `run_exp.sh`. You can simply change the training config to train different models.

### Key Files
- The model is implemented in https://github.com/taoyang1122/adapt-image-models/blob/main/mmaction/models/backbones/vit_clip.py. You may refer to it on how to apply AIM to your model.
- The weights are frozen at https://github.com/taoyang1122/adapt-image-models/blob/main/tools/train.py#L187.

## Evaluation
The code will do the evaluation after training. If you would like to evaluate a model only, please use the following command,
```shell
bash tools/dist_test.sh <PATH/TO/CONFIG> <CHECKPOINT_FILE> <NUM_GPU> --eval top_k_accuracy
```

Cheerfully, our work is under review. If you are interested in our work, please email to [jinchow21@sina.com](jinchow21@sina.com).


