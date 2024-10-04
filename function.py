import torch.nn.functional as F
import math
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
import cv2
from tqdm.auto import tqdm
import numpy as np
from torch import nn
import torch.optim as optim
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import os
import random
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image
import cv2
from tqdm.auto import tqdm
import numpy as np
from torch import nn
import torch.optim as optim
import pickle as pkl

def warp(feat, flow, mode='bilinear', padding_mode='zeros'):
    B, C, H, W = feat.size()
    rowv, colv = torch.meshgrid([torch.arange(0.5, H + 0.5, device=feat.device),
                                 torch.arange(0.5, W + 0.5, device=feat.device)])
    grid = torch.stack((colv, rowv), dim=0).unsqueeze(0).float()
    grid = grid + flow
    # 将grid映射到[-1,1]
    grid_norm_c = 2.0 * grid[:, 0] / W - 1.0
    grid_norm_r = 2.0 * grid[:, 1] / H - 1.0
    grid_norm = torch.stack((grid_norm_c, grid_norm_r), dim=1)
    grid_norm = grid_norm.permute(0, 2, 3, 1)
    # 使用grid_norm对原图进行采样
    output = F.grid_sample(feat, grid_norm, mode=mode, padding_mode=padding_mode,align_corners=True)
    return output

def get_activation(activation, activation_params=None, num_channels=None):
    if activation_params is None:
        activation_params = {}
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        print("error activation function！")
# 卷积块
def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, activation='relu', padding_mode='zeros', activation_params=None):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    activation_layer = get_activation(activation, activation_params, num_channels=out_planes)
    if activation_layer is not None:
        layers.append(activation_layer)
    return nn.Sequential(*layers)

def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    # Code from https://github.com/pytorch/pytorch/pull/5429/files
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = inizializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel


def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    """ Returns a 1-D Gaussian """
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


# 2维高斯函数
def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    """ Returns a 2-D Gaussian """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    if isinstance(sz, int):
        sz = (sz, sz)
    if isinstance(center, (list, tuple)):
        center = torch.tensor(center).view(1, 2)
    return gauss_1d(sz[0], sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * gauss_1d(
        sz[1], sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)

# 残差块
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=False, activation='relu',
                 padding_mode='zeros', attention='none'):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                                batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)
        self.conv2 = conv_block(planes, planes,  kernel_size=3, padding=1, dilation=dilation,
                                batch_norm=batch_norm, activation='none', padding_mode=padding_mode)
        self.downsample = downsample
        self.stride = stride
        self.activation = get_activation(activation, num_channels=planes)
        self.attention = None

    def forward(self, x):
        residual = x
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.attention is not None:
            out = self.attention(out)
        out += residual
        out = self.activation(out)
        return out
class PixShuffleUpsampler(nn.Module):
    def _get_gaussian_kernel(self, ksz, sd):
        assert ksz % 2 == 1
        K = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
        K = K / K.sum()
        return K

    def __init__(self, input_dim, output_dim, upsample_factor=2, use_bn=False, activation='relu',
                 icnrinit=False, gauss_blur_sd=None, gauss_ksz=3):
        super().__init__()
        pre_shuffle_dim = output_dim * upsample_factor ** 2
        self.conv_layer = conv_block(input_dim, pre_shuffle_dim, 1, stride=1, padding=0, batch_norm=use_bn,
                                     activation=activation, bias=not icnrinit)
        if icnrinit:
            # (https://arxiv.org/pdf/1707.02937.pdf) to reduce checkerboard artifacts
            # 减弱棋盘效应
            kernel = ICNR(self.conv_layer[0].weight, upsample_factor)
            self.conv_layer[0].weight.data.copy_(kernel)
        if gauss_blur_sd is not None:
            self.gauss_kernel = self._get_gaussian_kernel(gauss_ksz, gauss_blur_sd).unsqueeze(0)
        else:
            self.gauss_kernel = None
        # Pixelshuffle会将shape为( ∗ , r 2 C , H , W ) (*, r^2C, H, W)(∗,r 2C,H,W)的Tensor给reshape成( ∗ , C , r H , r W ) (*, C, rH,rW)(∗,C,rH,rW)的Tensor
        self.pix_shuffle = nn.PixelShuffle(upsample_factor)

    def forward(self, x):
        assert x.dim() == 4

        out = self.conv_layer(x)
        out = self.pix_shuffle(out)
        if self.gauss_kernel is not None:
            shape = out.shape
            out = out.view(-1, 1, *shape[-2:])
            gauss_ksz = getattr(self, 'gauss_ksz', 3)
            out = F.conv2d(out, self.gauss_kernel.to(out.device), padding=(gauss_ksz - 1) // 2)
            out = out.view(shape)
        return out
    
    
def get_gaussian_kernel(sd, ksz=None):
    """ Returns a 2D Gaussian kernel with standard deviation sd """
    if ksz is None:
        ksz = int(4 * sd + 1)

    assert ksz % 2 == 1
    K = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
    K = K / K.sum()
    return K.unsqueeze(0), ksz
def apply_kernel(im, ksz, kernel):
    """ apply the provided kernel on input image """
    shape = im.shape
    im = im.view(-1, 1, *im.shape[-2:])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode='reflect')
    im_out = F.conv2d(im, kernel).view(shape)
    return im_out
def match_colors(im_ref, im_q, im_test, ksz, gauss_kernel):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    gauss_kernel = gauss_kernel.to(im_ref.device)
    bi = 5

    # Apply Gaussian smoothing
    im_ref_mean = apply_kernel(im_ref, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()
    im_q_mean = apply_kernel(im_q, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()

    im_ref_mean_re = im_ref_mean.view(*im_ref_mean.shape[:2], -1)
    im_q_mean_re = im_q_mean.view(*im_q_mean.shape[:2], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        c = torch.lstsq(ir.t(), iq.t())
        c = c.solution[:3]
        c_mat_all.append(c)

    c_mat = torch.stack(c_mat_all, dim=0)
    im_q_mean_conv = torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_q_mean_conv = im_q_mean_conv.view(im_q_mean.shape)

    err = ((im_q_mean_conv - im_ref_mean) * 255.0).norm(dim=1)

    thresh = 20

    # If error is larger than a threshold, ignore these pixels
    valid = err < thresh

    pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    pad = [pad, pad, pad, pad]
    valid = F.pad(valid, pad)

    upsample_factor = im_test.shape[-1] / valid.shape[-1]
    valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear')
    valid = valid > 0.9

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    return im_t_conv, valid

class SpatialColorAlignment(nn.Module):
    def __init__(self, alignment_net, sr_factor=4):
        super().__init__()
        self.sr_factor = sr_factor
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.alignment_net.to(device)
        self.gauss_kernel = self.gauss_kernel.to(device)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear') * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear')

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz, self.gauss_kernel)

        return pred_warped_m, valid
    
# def unnormalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
#     """Unnormalize a tensor image with mean and standard deviation.

#     Args:
#         tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
#         mean (sequence): Sequence of means for each channel.
#         std (sequence): Sequence of standard deviations for each channel.
#         inplace(bool,optional): Bool to make this operation inplace.

#     Returns:
#         Tensor: Normalized Tensor image.
#     """
#     if not isinstance(tensor, torch.Tensor):
#         raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

#     if tensor.ndim < 3:
#         raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
#                          '{}.'.format(tensor.size()))

#     if not inplace:
#         tensor = tensor.clone()

#     dtype = tensor.dtype
#     mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
#     std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
#     if (std == 0).any():
#         raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
#     if mean.ndim == 1:
#         mean = mean.view(-1, 1, 1)
#     if std.ndim == 1:
#         std = std.view(-1, 1, 1)
#     tensor.mul_(std).add_(mean)
#     return tensor