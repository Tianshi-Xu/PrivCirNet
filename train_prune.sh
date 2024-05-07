# mbv2 SpENCNN, pruning
CUDA_VISIBLE_DEVICES=3 python train_prune.py -c configs/datasets/Prune/ConvNets/cifar10.yml --model c10_prune_mobilenetv2 /home/xts/code/dataset/cifar10/

CUDA_VISIBLE_DEVICES=7 python train_prune.py -c configs/datasets/Prune/ConvNets/cifar100.yml --model c100_prune_mobilenetv2 /home/xts/code/dataset/cifar100/

CUDA_VISIBLE_DEVICES=5 python train_prune.py -c configs/datasets/Prune/ConvNets/tiny.yml --model tiny_prune_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/

# vit SpENCNN, pruning
CUDA_VISIBLE_DEVICES=5 python train_prune.py -c configs/datasets/Prune/ViT/cifar10.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/

CUDA_VISIBLE_DEVICES=6 python train_prune.py -c configs/datasets/Prune/ViT/cifar100.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/

CUDA_VISIBLE_DEVICES=6 python train_prune.py -c configs/datasets/Prune/ViT/tiny.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/