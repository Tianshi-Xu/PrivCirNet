# vit
CUDA_VISIBLE_DEVICES=4 python train.py -c configs/datasets/ViT/tiny.yml --model vit_9_12_64_288 /home/xts/code/dataset/tiny-imagenet-200

# cvt
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/datasets/ViT/cifar10.yml --model cvt_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=2 python train.py -c configs/datasets/ViT/cifar100.yml --model cvt_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=3 python train.py -c configs/datasets/ViT/tiny.yml --model cvt_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# resnet18
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/datasets/ResNet/cifar10.yml --model cifar10_resnet18 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=2 python train.py -c configs/datasets/ResNet/cifar100.yml --model cifar100_resnet18 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=7 python train.py -c configs/datasets/ResNet/tiny.yml --model tiny_resnet18 /home/xts/code/dataset/tiny-imagenet-200/
