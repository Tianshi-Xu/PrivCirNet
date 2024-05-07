# mbv2 ILP
CUDA_VISIBLE_DEVICES=2 python CirILP.py -c configs/datasets/ConvNets/cifar10_ILP.yml --model c10_cir_mobilenetv2 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/ConvNets/cifar100_ILP.yml --model c100_cir_mobilenetv2 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/ConvNets/tiny_ILP.yml --model tiny_cir_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ConvNets/imagenet_ILP.yml --model image_cir_mobilenetv2 /opt/dataset/imagenet/

# vit ILP
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/ViT/cifar10_ILP.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ViT/cifar100_ILP.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/ViT/tiny_ILP.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# cvt ILP
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/CVT/cifar10_ILP.yml --model cvt_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/CVT/cifar100_ILP.yml --model cvt_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/CVT/tiny_ILP.yml --model cvt_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/
