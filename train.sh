# mbv2 fix
CUDA_VISIBLE_DEVICES=3 python train_cir.py -c configs/datasets/MBV2/cifar10_fix.yml --model c10_nas_mobilenetv2 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=5 python train_cir.py -c configs/datasets/MBV2/cifar100_fix.yml --model c100_nas_mobilenetv2 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=6 python train_cir.py -c configs/datasets/MBV2/tiny_fix.yml --model tiny_nas_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200

# vit fix
CUDA_VISIBLE_DEVICES=2 python train_cir.py -c configs/datasets/ViT/cifar10_fix.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=1 python train_cir.py -c configs/datasets/ViT/cifar100_fix.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=1 python train_cir.py -c configs/datasets/ViT/tiny_fix.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# mbv2 SpENCNN, pruning
CUDA_VISIBLE_DEVICES=1 python train_prune.py -c configs/datasets/Prune/cifar10.yml --model c10_prune_mobilenetv2 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=2 python train_prune.py -c configs/datasets/Prune/cifar100.yml --model c100_prune_mobilenetv2 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=3 python train_prune.py -c configs/datasets/Prune/tiny.yml --model tiny_prune_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/
