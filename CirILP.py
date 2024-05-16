#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pulp import LpVariable,LpProblem,LpInteger,LpMinimize,GLPK_CMD,LpStatus,value
import pulp
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import math
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
     model_parameters
from timm.models.layers import convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from src import *
from src.cir_layer import CirLinear,CirConv2d,CirBatchNorm2d
from src.utils.utils import KLLossSoft
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
# KD
parser.add_argument('--use-kd', action='store_true', default=False,
                    help='whether to use kd')
parser.add_argument('--kd-alpha', default=1.0, type=float,
                    help='KD alpha, soft loss portion (default: 1.0)')
parser.add_argument('--teacher', default='resnet101', type=str, metavar='MODEL',
                    help='Name of teacher model (default: "countception"')
parser.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize teacher model from this checkpoint (default: none)')
# CIRILP
parser.add_argument('--log_name', default='none', type=str,
                    help='act sparsification pattern')
parser.add_argument('--budget', default=1,
                    help='budget, before is avg block size, now is latency ratio')
parser.add_argument('--fix_blocksize',type=int, default=1,
                    help='whether to use dual skip for resnet blocks')
parser.add_argument('--better_initialization',type=bool, default=False,
                    help='better_initialization or not')
origin_latency = 0
rotate_mats = {}
rev_rotate_mats = {}
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def next_power_2(d):
    p = math.ceil(math.log2(d))
    return int(pow(2,p))

# cal_rot and cal_latency are used to compute latency of each layer, each block size
def cal_rot(n,m,d1,d2,b):
    # _logger.info("n,m,d1,d2,b: %d %d %d %d %d",n,m,d1,d2,b)
    min_rot = 1e8
    d_min = int(min(d2/b,d1/b))
    final_mp = 0
    final_d = 0
    for ri in range(1,(d_min)+1):
        for ro in range(1,(d_min)+1):
            d=int(ri*ro)
            m_p=int(n/b/d)
            if m*d_min*b<n:
                if d!=d_min:
                    continue
                i = 1
                while i<=m:
                    next_pow_2 = next_power_2(i*b)
                    if next_pow_2*d>n:
                        break
                    i+=1
                m_p=i-1
            if d>d_min or m_p>m or m_p<=0:
                continue
            if b!=1:
                next_pow_2 = next_power_2(m_p*b)
                if next_pow_2*d>n:
                    continue
            tmp=m*d1*(ri-1)/(m_p*b*d)+m*d2*(ro-1)/(m_p*b*d)
            if tmp<min_rot:
                min_rot=tmp
                final_d=d
                final_mp = m_p
    # _logger.info("final_mp,final_d: %d %d",final_mp,final_d)
    mul = math.ceil(1.0*m/final_mp)*math.ceil(1.0*d1/b/final_d)*math.ceil(1.0*d2/b/final_d)*final_d
    return min_rot, mul

def cal_latency(layer,HW,C,K,b,space,args):
    # _logger.info("HW,C,K,b: %d %d %d %d",HW,C,K,b)
    if space[-1]<b:
        b = space[-1]
    n=8192
    num_m=1
    if hasattr(layer,"padding") and layer.padding !=0:
        power2 = next_power_2(HW*b)
        if power2>n:
            num_m=int(math.ceil(power2/n))
            mp = n
        else:
            mp = power2
        n=int(math.floor(n/mp))
        C = C//b
        K = K//b
        b = 1
        HW=1
        # _logger.info("hw,n,C,K,b: %d %d %d %d %d",HW,n,C,K,b)
    rot, mul = cal_rot(n,HW,C,K,b)
    rot = rot*num_m
    mul = mul*num_m
    # _logger.info("rot, mul: %d %d",rot,mul)
    if "vit" in args.model:
        mul = HW*C*K/(n*b)
    return torch.tensor(rot+0.135*mul).item()

# compute |W'-W|^2
def cal_delta_w(layer,block_size,space,device):
    if space[-1]<block_size:
        block_size = space[-1]
    layer.fix_block_size = block_size
    cir_weight = layer.trans_to_cir(device)
    layer.fix_block_size = 1
    delta_w_2 = (torch.norm(layer.weight-cir_weight,p=2))**2
    return delta_w_2.item()

# compute the sensitivity $\Omega$
def cal_delta_fim_w(layer,block_size,space,device):
    if space[-1]<block_size:
        block_size = space[-1]
    layer.fix_block_size = block_size
    cir_weight = layer.trans_to_cir(device)

    layer.fix_block_size = 1
    # the grad need to be computed beforehand in training set
    delta_fim_w = torch.sum(((layer.weight-cir_weight)**2) * ((layer.weight.grad*1e5) **2))
    # _logger.info("grad.mean:"+str(torch.mean((layer.weight.grad*1e5) ** 2)))
    return delta_fim_w.item()

def create_teacher_model(args):
    teacher = create_model(
        args.teacher,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.,
        drop_connect_rate=0.,
        drop_block_rate=0.,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        # checkpoint_path=args.teacher_checkpoint,
        # checkpoint_path="",
    )
    if args.teacher_checkpoint != "":
        load_checkpoint(teacher, args.teacher_checkpoint, strict=True)
    teacher = teacher.eval()
    return teacher


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    handler = RotatingFileHandler(args.log_name+'.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)
    teacher = None
    if args.use_kd:
        teacher = create_teacher_model(args)
        teacher.cuda()
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_tf=args.bn_tf,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        fix_block_size=args.fix_blocksize,
        ILP=args.better_initialization)
    def _set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)
        
    if "vit" in args.model or "cvt" in args.model:
        for name,layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                hasBias = layer.bias is not None
                _set_module(model,name,CirLinear(layer.in_features,layer.out_features,args.fix_blocksize,hasBias,ILP=args.better_initialization))
    # print("OK1")
    if args.initial_checkpoint:
        load_checkpoint(model, args.initial_checkpoint,strict=False)
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    
    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats,download=True)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    train_loss_fn_kd = None
    if args.use_kd:    
        train_loss_fn_kd = KLLossSoft().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
    if args.initial_checkpoint != "":
        _logger.info("Verifying initial model in training dataset")
        # sample in the training set!
        train_metrics = train_one_epoch(
                0, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn,teacher=teacher,loss_fn_kd=train_loss_fn_kd)
    try:
        ILP(args,loader_eval,model)
    except KeyboardInterrupt:
        pass

# ILP function
def ILP(args,test_loader,model):
    target_block_size = args.budget
    for input, target in test_loader:
        break
    input, target = input.cuda(), target.cuda()
    device = input.device
    idx = 0
    origin_latency = 0
    # cir_idx used to store the index of cirLinear or cirConv2d
    cir_idx = []
    # search space is [1,2,4,8,16]
    delta_weights_b16 = []
    delta_weights_b8 = []
    delta_weights_b4 = []
    delta_weights_b2 = []
    delta_weights_b1 = []
    latency_weights_b16 = []
    latency_weights_b8 = []
    latency_weights_b4 = []
    latency_weights_b2 = []
    latency_weights_b1 = []
    sensitivity_b16 = []
    sensitivity_b8 = []
    sensitivity_b4 = []
    sensitivity_b2 = []
    sensitivity_b1 = []
    _logger.info("target_block_size:"+str(target_block_size))
    for layer in model.modules():
        if isinstance(layer, CirLinear) or isinstance(layer, CirConv2d):
            space = layer.search_space
            origin_latency += cal_latency(layer,layer.d1,layer.in_features,layer.out_features,target_block_size,space,args)
            cir_idx.append(idx)
            # delta_weights_b1 = |W'-W|^2
            delta_weights_b1.append(0)
            delta_weights_b2.append(cal_delta_w(layer,2,space,device))
            delta_weights_b4.append(cal_delta_w(layer,4,space,device))
            delta_weights_b8.append(cal_delta_w(layer,8,space,device))
            delta_weights_b16.append(cal_delta_w(layer,16,space,device))

            # sensitivity is $\Omega$ in our paper
            sensitivity_b1.append(0)
            sensitivity_b2.append(cal_delta_fim_w(layer,2,space,device))
            sensitivity_b4.append(cal_delta_fim_w(layer,4,space,device))
            sensitivity_b8.append(cal_delta_fim_w(layer,8,space,device))
            sensitivity_b16.append(cal_delta_fim_w(layer,16,space,device))
                  
            latency_weights_b1.append(cal_latency(layer,layer.d1,layer.in_features,layer.out_features,1,space,args))
            latency_weights_b2.append(cal_latency(layer,layer.d1,layer.in_features,layer.out_features,2,space,args))
            latency_weights_b4.append(cal_latency(layer,layer.d1,layer.in_features,layer.out_features,4,space,args))
            latency_weights_b8.append(cal_latency(layer,layer.d1,layer.in_features,layer.out_features,8,space,args))
            latency_weights_b16.append(cal_latency(layer,layer.d1,layer.in_features,layer.out_features,16,space,args))
            idx+=1
        elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            idx+=1
    # _logger.info(model)
    _logger.info("delta_weights_b1:"+str(delta_weights_b1))
    _logger.info("delta_weights_b2:"+str(delta_weights_b2))
    _logger.info("delta_weights_b4:"+str(delta_weights_b4))
    _logger.info("delta_weights_b8:"+str(delta_weights_b8))
    _logger.info("delta_weights_b16:"+str(delta_weights_b16))
    
    _logger.info("latency_weights_b1:"+str(latency_weights_b1))
    _logger.info("latency_weights_b2:"+str(latency_weights_b2))
    _logger.info("latency_weights_b4:"+str(latency_weights_b4))
    _logger.info("latency_weights_b8:"+str(latency_weights_b8))
    _logger.info("latency_weights_b16:"+str(latency_weights_b16))
    
    _logger.info("cir_idx:"+str(cir_idx))
    num_variable = len(cir_idx)
    variable = {}
    for i in range(num_variable):
        variable[f"b1_{i}"] = LpVariable(f"b1_{i}", 0, 1, cat=LpInteger)
        variable[f"b2_{i}"] = LpVariable(f"b2_{i}", 0, 1, cat=LpInteger)
        variable[f"b4_{i}"] = LpVariable(f"b4_{i}", 0, 1, cat=LpInteger)
        variable[f"b8_{i}"] = LpVariable(f"b8_{i}", 0, 1, cat=LpInteger)
        variable[f"b16_{i}"] = LpVariable(f"b16_{i}", 0, 1, cat=LpInteger)
        # variable[f"b32_{i}"] = LpVariable(f"b32_{i}", 0, 1, cat=LpInteger)
    prob = LpProblem("Block_size", LpMinimize)
    prob += sum(variable[f"b1_{i}"]*latency_weights_b1[i] +variable[f"b2_{i}"]*latency_weights_b2[i] + variable[f"b4_{i}"]*latency_weights_b4[i] +variable[f"b8_{i}"]*latency_weights_b8[i] +variable[f"b16_{i}"]*latency_weights_b16[i] for i in range(num_variable))-origin_latency <= 0.01

    
    #one layer only have one blocksize
    for i in range(num_variable):
        prob += (variable[f"b1_{i}"]+ variable[f"b2_{i}"]+ variable[f"b4_{i}"]+ variable[f"b8_{i}"]+ variable[f"b16_{i}"]) == 1
        
    delta_weights_b1 = np.array(delta_weights_b1)
    delta_weights_b2 = np.array(delta_weights_b2)
    delta_weights_b4 = np.array(delta_weights_b4)
    delta_weights_b8 = np.array(delta_weights_b8)
    delta_weights_b16 = np.array(delta_weights_b16)

    _logger.info("sensitivity_b1:"+str(sensitivity_b1))
    _logger.info("sensitivity_b2:"+str(sensitivity_b2))
    _logger.info("sensitivity_b4:"+str(sensitivity_b4))
    _logger.info("sensitivity_b8:"+str(sensitivity_b8))
    _logger.info("sensitivity_b16:"+str(sensitivity_b16))

    # optimization target: minimize the sensitivity
    prob += sum(variable[f"b1_{i}"]*sensitivity_b1[i] +variable[f"b2_{i}"]*sensitivity_b2[i] + variable[f"b4_{i}"]*sensitivity_b4[i] +variable[f"b8_{i}"]*sensitivity_b8[i] +variable[f"b16_{i}"]*sensitivity_b16[i] for i in range(num_variable))


    status = prob.solve(GLPK_CMD(msg=1, mip=1, options=["--tmlim", "10000","--simplex"]))
    
    LpStatus[status]

    result = []
    current_latency = 0
    for i in range(num_variable):
        block_size = value(variable[f"b1_{i}"])+value(variable[f"b2_{i}"])*2+value(variable[f"b4_{i}"])*4+value(variable[f"b8_{i}"])*8+value(variable[f"b16_{i}"])*16

        result.append(block_size)
        if block_size == 1:
            current_latency += latency_weights_b1[i]
        elif block_size == 2:
            current_latency += latency_weights_b2[i]
        elif block_size == 4:
            current_latency += latency_weights_b4[i]
        elif block_size == 8:
            current_latency += latency_weights_b8[i]
        elif block_size == 16:
            current_latency += latency_weights_b16[i]

    result = np.array(result)
    idx = 0
    for layer in model.modules():
        if isinstance(layer, CirLinear) or isinstance(layer, CirConv2d):
            if layer.search_space[-1] < result[idx]:
                result[idx] = layer.search_space[-1]
            idx += 1
    result_string = ','.join(str(item) for item in result)
    _logger.info("result: "+str(result_string))
    _logger.info("origin_latency:"+str(origin_latency))
    _logger.info("current_latency-origin_latency:"+str(current_latency-origin_latency))
    

# sample on training dataset and get gradients    
def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None,teacher=None,loss_fn_kd=None):
    
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    optimizer.zero_grad()
    total_samples = len(loader.dataset)
    loss = 0
    for batch_idx, (input, target) in enumerate(loader):
        if batch_idx>3000 and args.num_classes == 1000:
            total_samples = 3000
            break
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            if args.use_kd:
                target_t = teacher(input)
                loss = loss_fn_kd(output, target_t)
            else:
                loss = loss_fn(output, target)
        # _logger.info("loss:"+str(loss.item()))

        if args.num_classes == 1000 and batch_idx % 50==0:
            _logger.info("batch_idx:"+str(batch_idx))
        # for imagenet, number of samples are 2000
        
        # end for
        # break
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
        # print("------------------")
        # for param in model.parameters():
        #     if param.grad is not None:
        #         _logger.info("mean grad:"+str(torch.mean(param.grad.data)))
        # break
        
        
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    _logger.info("len loader.dataset:"+str(total_samples))
    # for param in model.parameters():
    #     if param.grad is not None:
    #         # _logger.info("mean grad:"+str(torch.mean(param.grad.data)))
    #         param.grad.data /= total_samples
    #         # _logger.info("mean grad2:"+str(torch.mean(param.grad.data)))
    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics



if __name__ == '__main__':
    main()
