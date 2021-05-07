import torch.nn as nn
import math

"""
Imported by https://github.com/marvis/pytorch-mobilenet/blob/master/main.py
"""


class MobileNetV1(nn.Module):
    def __init__(self, dropout_keep_prob=0.999):
        super(MobileNetV1, self).__init__()
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout = nn.Dropout(1 - dropout_keep_prob)
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
    
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            # truncated_normal_initializer in tensorflow
            nn.init.normal_(m.weight.data, std=0.09)
            #nn.init.constant(m.bias.data, 0)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x
