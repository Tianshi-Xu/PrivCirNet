import torch
import torch.nn as nn
import os
from .cir_layer import CirConv2d,CirBatchNorm2d
try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

def cir_conv3x3(in_planes, out_planes, stride=1, fix_block_size=1,ILP=False):
    """3x3 convolution with padding"""
    return CirConv2d(in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,fix_block_size=fix_block_size,ILP=ILP)


def cir_conv1x1(in_planes, out_planes, stride=1, fix_block_size=1,ILP=False):
    """1x1 convolution"""
    return CirConv2d(in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,fix_block_size=fix_block_size,ILP=ILP)


class CirBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        use_bn=True,
        use_relu=True,
        skip_last_relu=False,
        use_dual_skip=False,
        post_res_bn=False,
        fix_block_size=1,
        ILP=False
    ):
        super(CirBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = CirBatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("CirBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in CirBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = cir_conv3x3(inplanes, planes, stride,fix_block_size=fix_block_size,ILP=ILP)
        self.bn1 = norm_layer(planes, block_size=fix_block_size) if use_bn else nn.Identity()
        self.relu1 = (
            nn.ReLU(inplace=True) if use_relu else nn.PReLU(planes)
        )
        self.conv2 = cir_conv3x3(planes, planes, fix_block_size=fix_block_size, ILP=ILP)
        self.bn2 = norm_layer(planes, block_size=fix_block_size) if use_bn else nn.Identity()
        if skip_last_relu:
            self.relu2 = nn.Identity()
        else:
            self.relu2 = (
                nn.ReLU(inplace=True) if use_relu else nn.PReLU(planes)
            )
        self.downsample = downsample
        self.stride = stride
        self.use_dual_skip = use_dual_skip
        self.post_res_bn = post_res_bn

        # self.act_quant_op1 = TernaryFakeQuantize(is_trainable_weight=True)
        # self.act_quant_op2 = TernaryFakeQuantize(is_trainable_weight=True)

    def forward(self, x):
        identity = x

        out = x
        out = self.conv1(out)

        if not self.post_res_bn:
            out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.use_dual_skip:
            out += identity
            if self.post_res_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            identity = out
        else:
            if self.post_res_bn:
                out = self.bn1(out)
            out = self.relu1(out)

        out = self.conv2(out)

        if not self.post_res_bn:
            out = self.bn2(out)

        out += identity

        if self.post_res_bn:
            out = self.bn2(out)

        out = self.relu2(out)

        return out


class CirResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        use_bn=True,
        use_relu=True,
        skip_last_relu=False,
        down_block_type="default",
        use_dual_skip=False,
        post_res_bn=False,
        fix_block_size=1,
        ILP=False,
        **kwargs,
    ):
        super(CirResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.post_res_bn = post_res_bn
        use_relu=True
        # CIFAR10/CIFAR-100 (3x3 conv, stride=1)
        if num_classes == 10 or num_classes == 100 :
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Tiny (3x3 conv, stride=2)
        elif num_classes == 200:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        # ImageNet (7x7 conv, stride=2)
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # END

        # self.bn1 = norm_layer(self.inplanes) if use_bn else nn.Identity()
        self.bn1 = norm_layer(self.inplanes)
        self.relu = (
            nn.ReLU(inplace=True) if use_relu else nn.PReLU(self.inplanes)
        )
        self.layer1 = self._make_layer(
            block, 64, layers[0], use_bn=use_bn, use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
            use_dual_skip=use_dual_skip,
            post_res_bn=post_res_bn,fix_block_size=fix_block_size,ILP=ILP
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            use_bn=use_bn,
            use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
            use_dual_skip=use_dual_skip,
            post_res_bn=post_res_bn,fix_block_size=fix_block_size,ILP=ILP
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            use_bn=use_bn,
            use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
            use_dual_skip=use_dual_skip,
            post_res_bn=post_res_bn,fix_block_size=fix_block_size,ILP=ILP
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            use_bn=use_bn,
            use_relu=use_relu,
            skip_last_relu=skip_last_relu,
            down_block_type=down_block_type,
            use_dual_skip=use_dual_skip,
            post_res_bn=post_res_bn,fix_block_size=fix_block_size,ILP=ILP
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,CirConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm,CirBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CirBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
            use_bn=True, use_relu=True, skip_last_relu=False, down_block_type="default",
            use_dual_skip=False, post_res_bn=False,fix_block_size=1,ILP=False):
        norm_layer = CirBatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_block = {
                "default": [cir_conv1x1(self.inplanes, planes * block.expansion, stride,fix_block_size=fix_block_size,ILP=ILP)],
                "avgpool": [
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    cir_conv1x1(self.inplanes, planes * block.expansion, 1,fix_block_size=fix_block_size,ILP=ILP),
                ],
                "cir_conv3x3": [cir_conv3x3(self.inplanes, planes * block.expansion, stride,fix_block_size=fix_block_size,ILP=ILP)],
            }[down_block_type]
            downsample = nn.Sequential(
                *down_block,
                norm_layer(planes * block.expansion, block_size=fix_block_size)
                if use_bn and not post_res_bn else nn.Identity(),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                use_bn=use_bn,
                use_relu=use_relu,
                skip_last_relu=skip_last_relu,
                use_dual_skip=use_dual_skip,
                post_res_bn=post_res_bn,
                fix_block_size=fix_block_size,
                ILP=ILP
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_bn=use_bn,
                    use_relu=use_relu,
                    skip_last_relu=skip_last_relu,
                    post_res_bn=post_res_bn,
                    fix_block_size=fix_block_size,
                    ILP=ILP
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _cir_resnet(block, layers, fix_block_size=1, ILP=False, **kwargs):
    model = CirResNet(block, layers, fix_block_size=fix_block_size, ILP=ILP, **kwargs)
    return model


@register_model
def cir_cifar10_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print(str(kwargs))
    return _cir_resnet(
        CirBasicBlock, [2, 2, 2, 2], **kwargs
    )
    
@register_model
def cir_cifar100_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cir_resnet(
        CirBasicBlock, [2, 2, 2, 2], **kwargs
    )
    
@register_model
def cir_tiny_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _cir_resnet(
        CirBasicBlock, [2, 2, 2, 2], **kwargs
    )

if __name__ == '__main__':
    model = cir_cifar10_resnet18(fix_block_size=2,ILP=True)
    print(model)