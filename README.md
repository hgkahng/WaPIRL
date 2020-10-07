# Self-Supervised Learning for Wafer Bin Map Classification
This repository contains the original PyTorch implementation of the paper 'Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern Classification'.

## A. Requirements
- [Anaconda](https://www.anaconda.com/download/) (>= 4.8.3)
- [OpenCV](https://pypi.org/project/opencv-python/) (tested on 4.3.0.66)
- [PyTorch](https://pytorch.org) (tested on 1.6.0)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (tested on 0.8.5)
- [Albumentations](https://github.com/albumentations-team/albumentations) (tested on 0.4.6)
- [Tensorboard](https://www.tensorflow.org/tensorboard) (tested on 2.2.2, optional)
```
conda update -n base conda  # use 4.8.3 or higher
conda create -n wbm python=3.6
conda activate wbm
conda install anaconda
conda install opencv -c conda-forge
conda install pytorch=1.6.0 cudatoolkit=10.2 -c pytorch
pip install pytorch_lightning
pip install albumentations
```

## B. Dataset
1. Download from the following link: [WM-811K](https://www.kaggle.com/qingyi/wm811k-wafer-map)
2. Place the `LSWMD.pkl` file under `./data/wm811k/`.
3. Run the following script from the working directory:
```
python process_wm811k.py
```

## C. WaPIRL Pre-training
### C-1. From the command line (with default options)
```
python run_wapirl.py \
    --input_size 96 \
    --augmentation crop \
    --backbone_type resnet \
    --backbone_config 18 \
    --decouple_input \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 4 \
    --gpus 0 \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --weight_decay 1e-3 \
    --momentum 0.9 \
    --scheduler cosine \
    --warmup_steps 0 \
    --checkpoint_root ./checkpoints \
    --write_summary \
    --save_every 10 \
    --projector_type linear \
    --projector_size 128 \
    --temperature 0.07
```
- Run ```python run_wapirl.py --help``` for more information on arguments.
- If running on a Windows machine, set `num_workers` to 0. (multiprocessing does not function well.)

### C-2. From a configuration file
```
python run_wapirl.py @experiments/pretrain_wapirl.txt
```

## D. Fine-tuning
### D-1. From the command line (with default options)
```
python run_classification.py \
    --input_size 96 \
    --augmentation crop \
    --backbone_type resnet \
    --backbone_config 18 \
    --decouple_input \
    --epochs 100 \
    --batch_size 256 \
    --num_workers 4 \
    --gpus 0 \
    --optimizer sgd \
    --learning_rate 1e-2 \
    --weight_decay 1e-3 \
    --momentum 0.9 \
    --scheduler cosine \
    --warmup_steps 0 \
    --checkpoint_root ./checkpoints \
    --write_summary \
    --pretrained_model_file /path/to/file \
    --pretrained_model_type wapirl \
    --label_proportion 1.00 \
    --label_smoothing 0.1 \
    --dropout 0.5
```
- IMPORTANT: Provide the correct path for the `pretrained_model_file` argument.
- Run ```python run_classification.py --help``` for more information on arguments.
- If running on a Windows machine, set `num_workers` to 0. (multiprocessing does not function well.)

### D-2. From a configuration file
```
python run_classification.py @experiments/finetune_wapirl.txt
```
