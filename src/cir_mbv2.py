import torch.nn as nn
import torch
import math
from .cir_layer import CirConv2d,CirBatchNorm2d
try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class CirInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,fix_block_size=-1,ILP=False):
        super(CirInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.fix_block_size = fix_block_size
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                CirConv2d(inp, hidden_dim, 1, 1,fix_block_size=fix_block_size,ILP=ILP),
                CirBatchNorm2d(hidden_dim,block_size=fix_block_size),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                CirConv2d(hidden_dim, oup, 1, 1,fix_block_size=fix_block_size,ILP=ILP),
                CirBatchNorm2d(oup,block_size=fix_block_size),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class CirMobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0,fix_block_size=-1,ILP=False):
        super(CirMobileNetV2, self).__init__()
        block = CirInvertedResidual
        input_channel = 32
        last_channel = 1280
        if input_size == 224:
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        else:
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        if n_class != 10 and n_class != 100:
            self.features = [conv_bn(3, input_channel, 2)]
        else:
            self.features = [conv_bn(3, input_channel, 1)]
        # building inverted residual blocks
        idx=0
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,fix_block_size=fix_block_size,ILP=ILP))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,fix_block_size=fix_block_size,ILP=ILP))
                input_channel = output_channel
            idx+=1
        # building last several layers
        self.features = nn.Sequential(*self.features)
        self.conv = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
def cir_nas_mobilenet(n_class, input_size, width_mult,fix_block_size=-1,ILP=False) -> CirMobileNetV2:
    model = CirMobileNetV2(n_class=n_class, input_size=input_size, width_mult=width_mult,fix_block_size=fix_block_size)
    return model

@register_model
def image_nas_mobilenetv2(fix_block_size=-1,ILP=False,**kwargs):
    model=cir_nas_mobilenet(1000,224,1,fix_block_size=fix_block_size,ILP=ILP)
    return model

@register_model
def tiny_nas_mobilenetv2(fix_block_size=-1,ILP=False,**kwargs):
    model=cir_nas_mobilenet(200,64,1,fix_block_size=fix_block_size,ILP=ILP)
    return model

@register_model
def c100_nas_mobilenetv2(fix_block_size=-1,ILP=False,**kwargs):
    model=cir_nas_mobilenet(100,32,1,fix_block_size=fix_block_size,ILP=ILP)
    return model

@register_model
def c10_nas_mobilenetv2(fix_block_size=-1,ILP=False,**kwargs):
    model=cir_nas_mobilenet(10,32,1,fix_block_size=fix_block_size,ILP=ILP)
    return model
