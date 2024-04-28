# mbv2 ILP
CUDA_VISIBLE_DEVICES=3 python CirILP.py -c configs/datasets/MBV2/cifar10_ILP.yml --model c10_nas_mobilenetv2 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=3 python CirILP.py -c configs/datasets/MBV2/cifar100_ILP.yml --model c100_nas_mobilenetv2 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=1 python CirILP.py -c configs/datasets/MBV2/tiny_ILP.yml --model tiny_nas_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/MBV2/imagenet_ILP.yml --model image_nas_mobilenetv2 /opt/dataset/imagenet/

# vit ILP
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/ViT/cifar10_ILP.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/ViT/cifar100_ILP.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ViT/tiny_ILP.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200