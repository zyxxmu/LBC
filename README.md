# Learning Best Combination for Efficient N:M Sparsity

Pytorch implementation of our paper "Learning Best Combination for Efficient N:M Sparsity" ([Link](https://arxiv.org/abs/2206.06662))

## Data Preparation

- The ImageNet dataset should be prepared as follows:

```text
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
```

## Requirements

- python 3.7
- pytorch 1.10.2
- torchvision 0.11.3

## Re-produce our results

- ResNet on ImageNet

```bash
cd ResNet
python imagenet.py --job_dir PATH_TO_JOB_DIR --t_i 0 --t_f 60 --gpus 0 1 2 3 --train_batch_size 256 --eval_batch_size 256 --lr 0.1 --label-smoothing 0.1 --N 2 --M 4 --data_path PATH_TO_DATASETS
```

- DeiT-small on ImageNet

```bash
cd DeiT-small
python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir PATH_TO_JOB_DIR
```

Besides, we provide our trained models and experiment logs at [Google Drive](https://drive.google.com/drive/folders/1PYuZUgeI9Mz7yfeD3lartUcZF7Zo4u6-?usp=sharing). To test, run:

- ResNet on ImageNet

```bash
cd ResNet
python eval.py --pretrain_dir PATH_TO_CHECKPOINTS --gpus 0 --train_batch_size 256 --eval_batch_size 256  --label-smoothing 0.1 --N 2 --M 4 --data_path PATH_TO_DATASETS
```

- DeiT-small on ImageNet

```bash
cd DeiT-small
python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir PATH_TO_JOB_DIR --resume PATH_TO_CHECKPOINTS --eval
```

