# mbv2 SpENCNN, pruning
CUDA_VISIBLE_DEVICES=3 python train_prune.py -c configs/datasets/Prune/MBV2/cifar10.yml --model c10_prune_mobilenetv2 /home/xts/code/dataset/cifar10/

CUDA_VISIBLE_DEVICES=7 python train_prune.py -c configs/datasets/Prune/MBV2/cifar100.yml --model c100_prune_mobilenetv2 /home/xts/code/dataset/cifar100/

CUDA_VISIBLE_DEVICES=5 python train_prune.py -c configs/datasets/Prune/MBV2/tiny.yml --model tiny_prune_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/

CUDA_VISIBLE_DEVICES=4 python train_prune.py -c configs/datasets/Prune/MBV2/imagenet.yml --model image_prune_mobilenetv2 /opt/dataset/imagenet/


# vit SpENCNN, pruning
CUDA_VISIBLE_DEVICES=5 python train_prune.py -c configs/datasets/Prune/ViT/cifar10.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/

CUDA_VISIBLE_DEVICES=6 python train_prune.py -c configs/datasets/Prune/ViT/cifar100.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/

CUDA_VISIBLE_DEVICES=6 python train_prune.py -c configs/datasets/Prune/ViT/tiny.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# cvt SpENCNN, pruning
CUDA_VISIBLE_DEVICES=1 python train_prune.py -c configs/datasets/Prune/CVT/cifar10.yml --model cvt_7_4_32 /home/xts/code/dataset/cifar10/

CUDA_VISIBLE_DEVICES=2 python train_prune.py -c configs/datasets/Prune/CVT/cifar100.yml --model cvt_7_4_32_c100 /home/xts/code/dataset/cifar100/

CUDA_VISIBLE_DEVICES=3 python train_prune.py -c configs/datasets/Prune/CVT/tiny.yml --model cvt_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# resnet SpENCNN, pruning
CUDA_VISIBLE_DEVICES=2 python train_prune.py -c configs/datasets/Prune/ResNet/cifar10.yml --model prune_cifar10_resnet18 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=3 python train_prune.py -c configs/datasets/Prune/ResNet/cifar100.yml --model prune_cifar100_resnet18 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=4 python train_prune.py -c configs/datasets/Prune/ResNet/tiny.yml --model prune_tiny_resnet18 /home/xts/code/dataset/tiny-imagenet-200/
