# PrivCirNet: Efficient Private Inference via Block Circulant Transformation

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2405.14569)&nbsp;
</div>

#### 🔥 PrivCirNet is accepted by NeurIPS'24!!
#### 🔥 Introducing a network-protocol co-optimization framework for accelerating homomorphic encrypted network inference.

## PrivCirNet zoo
We provide PrivCirNet models which can be downloaded [here](https://drive.google.com/drive/folders/17vV8wySe9fMYi0vgGlHUhuWZtKpW59HC?usp=sharing).

## Installation

1. Install `torch>=2.0.0`, `python>=3.10`.
2. Install other pip packages via `pip install -r requirements.txt`.
3. Prepare the dataset including CIFAR-10, CIFAR100, Tiny ImageNet and ImageNet.

## Training Scripts
### Run CirILP.py to get block size configuration
In this step, we load the pretrained model and get the layer-wise block sizes configuration. The script is `ILP.sh`. For example, to configure ViT on CIFAR-10, you can run the following command:
```shell
CUDA_VISIBLE_DEVICES=1 python CirILP.py -c configs/datasets/ViT/cifar10_ILP.yml --model vit_7_4_32 PATH_TO_CIFAR10
```
A log file named `vit_c10_ILP.log` will be created to save the configuration and logs. You will get the layer-wise block sizes configuration in the log file.

There are two parameters in the `.yml` file you need to specify:
- budget. You can set the budget to `2`, `4`, and `8` whose latency is less than that of networks with uniform block size `2`, `4`, and `8`.
- better_initialization. `True` means use the initialization method proposed in PrivCirNet, `False` means use the previous initialization method, i.e., $\min |W'-W|_2^2$.

### Train the circulant models
In this step, we load the pretrained model and train the circulant models with the layer-wise block sizes configuration obtained from the previous step. 

This step is simple which is the same as training the original models. The script is `train_cir.sh`. For example , to train ViT on CIFAR-10, you can run the following command:
```shell
CUDA_VISIBLE_DEVICES=1 python train_cir.py -c configs/datasets/ViT/cifar10_fix.yml --model vit_7_4_32 PATH_TO_CIFAR10
```
A log file named `your_log_name.log` will be created to save logs. The checkpoints will be saved in the `output` folder. There are several parameters in the `.yml` file you need to specify:
- `fix_blocksize_list`. It is the block size configuration you get from the previous step. It must be separated by commas.
- `log_name`. The name of the log file.
- `use_kd`. Whether to use knowledge distillation. If `True`, you need to specify the `teacher_model` and `teacher_checkpoint` in the `.yml` file.
- `initial_checkpoint`. You must load the initial checkpoint from the original model. All initial checkpoints can be downloaded [here](https://drive.google.com/drive/folders/18R8JJ-FGkNd8m6TXdBcV52H0bM5LYkRr?usp=sharing).

### Train other baseline models
This repository also provides the training scripts for the two baselines:
- Uniform block size circulant networks. The training script is the same as `Train the circulant models` where you can set `fix_blocksize` to `2`, `4`, and `8` to train uniform block size circulant networks.
- SpENCNN (structured pruning). The training script is `train_prune.sh`. For example, to train MobileNetV2 on CIFAR-10, you can run the following command:
```shell
CUDA_VISIBLE_DEVICES=1 python train_prune.py -c configs/datasets/Prune/cifar10.yml --model c10_prune_mobilenetv2 PATH_TO_CIFAR10
```
- You can set `prune_ratio` in the `.yml` file to specify the pruning ratio.
- Pruning models can be downloaded [here](https://drive.google.com/drive/folders/1tr-pDDPkFIpb_jCB267W3RUIqylCF0Jn?usp=sharing).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@misc{xu2024privcirnetefficientprivateinference,
      title={PrivCirNet: Efficient Private Inference via Block Circulant Transformation}, 
      author={Tianshi Xu and Lemeng Wu and Runsheng Wang and Meng Li},
      year={2024},
      eprint={2405.14569},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2405.14569}, 
}
```