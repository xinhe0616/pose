import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

from config import config
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion,
#                                   momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
class PoseResNet(nn.Module):

    def __init__(self, block, layers, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.temporal = config.temporal
        # lstm
        self.outclass = 13
        self.conv_ix_lstm = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        # initial lstm
        self.conv_gx_lstm0 = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(256 + self.outclass, 256, kernel_size=3, padding=1)



        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=13,
            kernel_size=1,
            stride=1,
            padding=1 if 1 == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #
    #     x = self.deconv_layers(x)
    #     x = self.final_layer(x)
    #     return x
    def lstm(self, heatmap, features, hide_t_1, cell_t_1):
        '''
        :param heatmap:     (class+1) * 45 * 45
        :param features:    32 * 45 * 45
        :param centermap:   1 * 45 * 45
        :param hide_t_1:    48 * 45 * 45
        :param cell_t_1:    48 * 45 * 45
        :return:
        hide_t:    48 * 45 * 45
        cell_t:    48 * 45 * 45
        '''
        xt = torch.cat([heatmap, features], dim=1)  # (32+ class+1 +1 ) * 45 * 45

        gx = self.conv_gx_lstm(xt)  # output: 48 * 45 * 45
        gh = self.conv_gh_lstm(hide_t_1)  # output: 48 * 45 * 45
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)  # output: 48 * 45 * 45
        oh = self.conv_oh_lstm(hide_t_1)  # output: 48 * 45 * 45
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)  # output: 48 * 45 * 45
        ih = self.conv_ih_lstm(hide_t_1)  # output: 48 * 45 * 45
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)  # output: 48 * 45 * 45
        fh = self.conv_fh_lstm(hide_t_1)  # output: 48 * 45 * 45
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)

        return cell_t, hide_t

    def lstm0(self, x):
        gx = self.conv_gx_lstm0(x)
        ix = self.conv_ix_lstm0(x)
        ox = self.conv_ox_lstm0(x)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell1 = torch.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1

    def _resnet1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def _resnet2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        return x

    def _resnet3(self, x):

        x = self.final_layer(x)
        return x
    # def forward(self, images):
    #     heat_maps = []
    #     image0 = images[:,0]
    #     initial_heatmap = self._resnet1(image0)
    #     feture = self._resnet2(image0)
    #     x = torch.cat([initial_heatmap, feture], dim=1)
    #     cell, hide = self.lstm0(x)
    #     heatmap = self._resnet3(hide)
    #     heat_maps.append(initial_heatmap)  # for initial loss
    #     heat_maps.append(heatmap)
    #
    #     for i in range(1,5):
    #         image = images[:,i]
    #         feature = self._resnet2(image)
    #         cell, hide = self.lstm0(heatmap, feature)
    #         heatmap = self._resnet3(hide)
    #         heat_maps.append(heatmap)
    #     return heat_maps
    def forward(self, images):
        heat_maps = []
        for i in range(self.temporal):
            image = images[:,i]
            heatmap = self._resnet1(image)
            heat_maps.append(heatmap)
        return heat_maps

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3])
    # ,
    #            50: (Bottleneck, [3, 4, 6, 3]),
    #            101: (Bottleneck, [3, 4, 23, 3]),
    #            152: (Bottleneck, [3, 8, 36, 3])
               }




import torchvision
def get_pose_net():
    num_layers = 18
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers)
    return model
