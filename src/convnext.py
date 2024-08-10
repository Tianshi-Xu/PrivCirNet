import sys

import torch
import torch.nn as nn

from timm.models import ConvNeXt
try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model
from timm.models.helpers import build_model_with_cfg


def _create_convnext_cifar(
    variant, pretrained=False, **kwargs
):
    drop_key = ['drop_connect_rate', 'drop_block_rate', 'bn_tf', 'fix_block_size', 'ILP']
    for key in drop_key:
        if key in kwargs:
            kwargs.pop(key)
    model = build_model_with_cfg(
        ConvNeXt, variant, pretrained,
        # pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


@register_model
def convnext_cifar_v1(pretrained=False, **kwargs):
    model_args = dict(
        depths=(2, 2, 2, 2),
        dims=(80, 160, 320, 640),
        mlp_ratios=(4, 4, 4, 4),
        head_norm_first=True,
        conv_mlp=True,
        stem_type="patch",
        stem_kernel_size=1,
        stem_stride=1,
        **kwargs)
    model = _create_convnext_cifar(
        'convnext_nano_hnf',
        pretrained=pretrained,
        **model_args)
    return model


@register_model
def convnext_cifar_nano_hnf(pretrained=False, **kwargs):
    print(*kwargs)
    model_args = dict(
        depths=(2, 2, 8, 2),
        dims=(80, 160, 320, 640),
        head_norm_first=True,
        conv_mlp=True,
        stem_type="patch",
        stem_kernel_size=3,
        stem_stride=1,
        **kwargs)
    num_classes = kwargs.get('num_classes')
    model = _create_convnext_cifar(
        'convnext_nano_hnf',
        **model_args)
    return model


@register_model
def convnext_cifar_tiny_hnf(pretrained=False, **kwargs):
    model_args = dict(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        head_norm_first=True,
        conv_mlp=True,
        stem_type="patch",
        stem_kernel_size=1,
        stem_stride=1,
        **kwargs)
    model = _create_convnext_cifar(
        'convnext_tiny_hnf',
        pretrained=pretrained,
        **model_args)
    return model
