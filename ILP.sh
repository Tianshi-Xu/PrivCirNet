# mbv2 ILP
CUDA_VISIBLE_DEVICES=2 python CirILP.py -c configs/datasets/MBV2/cifar10_ILP.yml --model c10_cir_mobilenetv2 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/MBV2/cifar100_ILP.yml --model c100_cir_mobilenetv2 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/MBV2/tiny_ILP.yml --model tiny_cir_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/
CUDA_VISIBLE_DEVICES=4 python CirILP.py -c configs/datasets/MBV2/imagenet_ILP.yml --model image_cir_mobilenetv2 /opt/dataset/imagenet/

# resnet ILP
CUDA_VISIBLE_DEVICES=1 python CirILP.py -c configs/datasets/ResNet/cifar10_ILP.yml --model cir_cifar10_resnet18 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=2 python CirILP.py -c configs/datasets/ResNet/cifar100_ILP.yml --model cir_cifar100_resnet18 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=3 python CirILP.py -c configs/datasets/ResNet/tiny_ILP.yml --model cir_tiny_resnet18 /home/xts/code/dataset/tiny-imagenet-200/

# vit ILP
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/ViT/cifar10_ILP.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=7 python CirILP.py -c configs/datasets/ViT/cifar100_ILP.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=3 python CirILP.py -c configs/datasets/ViT/tiny_ILP.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# cvt ILP
CUDA_VISIBLE_DEVICES=4 python CirILP.py -c configs/datasets/CVT/cifar10_ILP.yml --model cvt_7_4_32 /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/CVT/cifar100_ILP.yml --model cvt_7_4_32_c100 /home/xts/code/dataset/cifar100/
CUDA_VISIBLE_DEVICES=6 python CirILP.py -c configs/datasets/CVT/tiny_ILP.yml --model cvt_9_12_64 /home/xts/code/dataset/tiny-imagenet-200/

# convnext 

CUDA_VISIBLE_DEVICES=4 python CirILP.py -c configs/datasets/ConvNeXt/cifar10_ILP.yml --model convnext_cifar_nano_hnf /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=5 python CirILP.py -c configs/datasets/ConvNeXt/cifar100_ILP.yml --model convnext_cifar_nano_hnf /home/xts/code/dataset/cifar100/

# deepreshape
CUDA_VISIBLE_DEVICES=2 python CirILP.py -c configs/datasets/DeepReshape/cifar100_ILP.yml --model cir_cifar100_resnet18_553 /home/xts/code/dataset/cifar100/

CUDA_VISIBLE_DEVICES=2 python CirILP.py -c configs/datasets/DeepReshape/cifar100_ILP_253.yml --model cir_cifar100_resnet18_253 /home/xts/code/dataset/cifar100/

# regnet
CUDA_VISIBLE_DEVICES=1 python CirILP.py -c configs/datasets/RegNet/cifar10_ILP.yml --model RegNetX_200MF /home/xts/code/dataset/cifar10/
CUDA_VISIBLE_DEVICES=1 python CirILP.py -c configs/datasets/RegNet/cifar100_ILP.yml --model RegNetX_200MF /home/xts/code/dataset/cifar100/
