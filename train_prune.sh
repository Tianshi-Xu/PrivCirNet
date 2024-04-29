# mbv2 SpENCNN, pruning
CUDA_VISIBLE_DEVICES=7 python train_prune.py -c configs/datasets/Prune/cifar10.yml --model c10_prune_mobilenetv2 PATH_TO_CIFAR10
CUDA_VISIBLE_DEVICES=7 python train_prune.py -c configs/datasets/Prune/cifar100.yml --model c100_prune_mobilenetv2 PATH_TO_CIFAR100
CUDA_VISIBLE_DEVICES=5 python train_prune.py -c configs/datasets/Prune/tiny.yml --model tiny_prune_mobilenetv2 PATH_TO_TINYIMAGENET
