# mbv2 ILP
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/MBV2/cifar10_ILP.yml --model c10_cir_mobilenetv2 PATH_TO_CIFAR10
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/MBV2/cifar100_ILP.yml --model c100_cir_mobilenetv2 PATH_TO_CIFAR100
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/MBV2/tiny_ILP.yml --model tiny_cir_mobilenetv2 PATH_TO_TINYIMAGENET
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/MBV2/imagenet_ILP.yml --model image_cir_mobilenetv2 PATH_TO_IMAGENET

# vit ILP
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ViT/cifar10_ILP.yml --model vit_7_4_32 PATH_TO_CIFAR10
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ViT/cifar100_ILP.yml --model vit_7_4_32_c100 PATH_TO_CIFAR100
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ViT/tiny_ILP.yml --model vit_9_12_64 PATH_TO_TINYIMAGENET
