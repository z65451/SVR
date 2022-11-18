import torchvision
import skimage
import numpy as np
import kornia
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam
import numpy
from einops import rearrange
import time

from zmq import device
from transformer import Transformer
from Intra_MLP import index_points, knn_l2
import torchvision.transforms.functional as fn
import itertools
import cv2
from torch.nn import functional as f
from VAE.models import BaseVAE
from VAE.models.types_ import *
from VAE.models.vanilla_vae import VanillaVAE

from torch.autograd import Variable

from AE import Network as Network2


import sys
sys.path.append('NonUniformBlurKernelEstimation/')

from NonUniformBlurKernelEstimation.models.TwoHeadsNetwork import TwoHeadsNetwork
def dilation_pytorch(image, strel, origin=(0, 0), border_value=0):
    # first pad the image to have correct unfolding; here is where the origins is used
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1],
                      strel.shape[1] - origin[1] - 1], mode='constant', value=border_value)
    # Unfold the image to be able to perform operation on neighborhoods
    image_unfold = f.unfold(image_pad.unsqueeze(
        0).unsqueeze(0), kernel_size=strel.shape)
    # Flatten the structural element since its two dimensions have been flatten when unfolding
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    # Perform the greyscale operation; sum would be replaced by rest if you want erosion
    sums = image_unfold + strel_flatten
    # Take maximum over the neighborhood
    result, _ = sums.max(dim=1)
    # Reshape the image to recover initial shape
    return torch.reshape(result, image.shape)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


def torch_gather_nd5(params: torch.Tensor,
                     indices: torch.Tensor) -> torch.Tensor:
    """
    Perform the tf.gather_nd on torch.Tensor. Although working, this implementation is
    quite slow and 'ugly'. You should not care to much about performance when using
    this function. I encourage you to think about how to optimize this.

    This function has not been tested properly. It has only been tested empirically
    and shown to yield identical results compared to tf.gather_nd. Still, use at your
    own risk.

    Does not support the `batch_dims` argument that tf.gather_nd does support. This is
    something for future work.

    :param params: (Tensor) - the source Tensor
    :param indices: (LongTensor) - the indices of elements to gather
    :return output: (Tensor) – the destination tensor
    """
    assert indices.dtype == torch.int64, f"indices must be torch.LongTensor, got {indices.dtype}"
    assert indices.shape[-1] <= len(params.shape), f'The last dimension of indices can be at most the rank ' \
                                                   f'of params ({len(params.shape)})'

    # Define the output shape. According to the  documentation of tf.gather_nd, this is:
    # "indices.shape[:-1] + params.shape[indices.shape[-1]:]"
    output_shape = indices.shape[:-1] + params.shape[indices.shape[-1]:]

    # Initialize the output Tensor as an empty one.
    output = torch.zeros(
        size=output_shape, device=params.device, dtype=params.dtype)

    # indices_to_fill is a list of tuple containing the indices to fill in `output`
    indices_to_fill = list(itertools.product(
        *[range(x) for x in output_shape[:-1]]))

    # Loop over the indices_to_fill and fill the `output` Tensor
    for idx in indices_to_fill:
        index_value = indices[idx]

        if len(index_value.shape) == 0:
            index_value = torch.Tensor([0, index_value.item()])

        value = params[index_value.view(-1, 1).tolist()].view(-1)
        output[idx] = value

    return output


# vgg choice
base = {'vgg': [64, 64, 'M', 128, 128, 'M', 256, 256,
                256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}



def gather_nd4(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


def gather_nd(params, indices, batch_dims=0):
    """ The same as tf.gather_nd.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    if isinstance(indices, torch.Tensor):
        indices = indices.numpy()
    else:
        if not isinstance(indices, np.array):
            raise ValueError(
                f'indices must be `torch.Tensor` or `numpy.array`. Got {type(indices)}')
    if batch_dims == 0:
        orig_shape = list(indices.shape)
        num_samples = int(np.prod(orig_shape[:-1]))
        m = orig_shape[-1]
        n = len(params.shape)

        if m <= n:
            out_shape = orig_shape[:-1] + list(params.shape[m:])
        else:
            raise ValueError(
                f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
            )
        indices = indices.reshape((num_samples, m)).transpose().tolist()
        output = params[indices]    # (num_samples, ...)
        return output.reshape(out_shape).contiguous()
    else:
        batch_shape = params.shape[:batch_dims]
        orig_indices_shape = list(indices.shape)
        orig_params_shape = list(params.shape)
        assert (
            batch_shape == indices.shape[:batch_dims]
        ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
        mbs = np.prod(batch_shape)
        if batch_dims != 1:
            params = params.reshape(mbs, *(params.shape[batch_dims:]))
            indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
        output = []
        for i in range(mbs):
            output.append(gather_nd(params[i], indices[i], batch_dims=0))
        output = torch.stack(output, dim=0)
        output_shape = orig_indices_shape[:-1] + \
            list(orig_params_shape[orig_indices_shape[-1]+batch_dims:])
        return output.reshape(*output_shape).contiguous()
# vgg16


def vgg(cfg, i=3, batch_norm=True):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return layers


def hsp(in_channel, out_channel):
    layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                           nn.ReLU(inplace=False))
    return layers


def cls_modulation_branch(in_channel, hiden_channel):
    layers = nn.Sequential(nn.Linear(in_channel, hiden_channel),
                           nn.ReLU(inplace=False))
    return layers


def cls_branch(hiden_channel, class_num):
    layers = nn.Sequential(nn.Linear(hiden_channel, class_num),
                           nn.Sigmoid())
    return layers


def intra():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.Sigmoid()]
    return layers


def concat_r():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU(inplace=False)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU(inplace=False)]
    layers += [nn.ConvTranspose2d(512, 512, 4, 2, 1)]
    return layers


def concat_1():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU(inplace=False)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU(inplace=False)]
    return layers


def mask_branch():
    layers = []
    layers += [nn.Conv2d(512, 2, 3, 1, 1)]
    layers += [nn.ConvTranspose2d(2, 2, 8, 4, 2)]
    layers += [nn.Sigmoid()]
    return layers


def incr_channel():
    layers = []
    layers += [nn.Conv2d(128, 512, 3, 1, 1)]
    layers += [nn.Conv2d(256, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    return layers


def incr_channel2():
    layers = []
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU(inplace=False)]
    return layers


def norm(x, dim):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    normed = x / torch.sqrt(squared_norm)
    return normed


def fuse_hsp(x, p, group_size=5):

    t = torch.zeros(group_size, x.size(1))
    for i in range(x.size(0)):
        tmp = x[i, :]
        if i == 0:
            nx = tmp.expand_as(t)
        else:
            nx = torch.cat(([nx, tmp.expand_as(t)]), dim=0)
    nx = nx.view(x.size(0)*group_size, x.size(1), 1, 1)
    y = nx.expand_as(p)
    return y


def warp(x, flo):

    # x: [B, C, H, W] (im2)
    # flo: [B, 2, H, W] flow

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    flo = flo.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output*mask


def gather_nd2(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """
    Return coordinate matrices from coordinate vectors.
    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.
    .. versionchanged:: 1.9
       1-D and 0-D cases are allowed.
    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
        .. versionadded:: 1.7.0
    sparse : bool, optional
        If True the shape of the returned coordinate array for dimension *i*
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
        intended to be use with :ref:`basics.broadcasting`.  When all
        coordinates are used in an expression, broadcasting still leads to a
        fully-dimensonal result array.
        Default is False.
        .. versionadded:: 1.7.0
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.
        .. versionadded:: 1.7.0
    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,
        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.
    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::
        xv, yv = np.meshgrid(x, y, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]
        xv, yv = np.meshgrid(x, y, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]
    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.
    See Also
    --------
    mgrid : Construct a multi-dimensional "meshgrid" using indexing notation.
    ogrid : Construct an open multi-dimensional "meshgrid" using indexing
            notation.
    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = np.meshgrid(x, y)
    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])
    >>> yv
    array([[0.,  0.,  0.],
           [1.,  1.,  1.]])
    >>> xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[0. ,  0.5,  1. ]])
    >>> yv
    array([[0.],
           [1.]])
    `meshgrid` is very useful to evaluate functions on a grid.  If the
    function depends on all coordinates, you can use the parameter
    ``sparse=True`` to save memory and computation time.
    >>> x = np.linspace(-5, 5, 101)
    >>> y = np.linspace(-5, 5, 101)
    >>> # full coordinate arrays
    >>> xx, yy = np.meshgrid(x, y)
    >>> zz = np.sqrt(xx**2 + yy**2)
    >>> xx.shape, yy.shape, zz.shape
    ((101, 101), (101, 101), (101, 101))
    >>> # sparse coordinate arrays
    >>> xs, ys = np.meshgrid(x, y, sparse=True)
    >>> zs = np.sqrt(xs**2 + ys**2)
    >>> xs.shape, ys.shape, zs.shape
    ((1, 101), (101, 1), (101, 101))
    >>> np.array_equal(zz, zs)
    True
    >>> import matplotlib.pyplot as plt
    >>> h = plt.contourf(x, y, zs)
    >>> plt.axis('scaled')
    >>> plt.colorbar()
    >>> plt.show()
    """
    ndim = len(xi)

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
              for i, x in enumerate(xi)]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = np.broadcast_arrays(*output, subok=True)

    if copy:
        output = [x.copy() for x in output]

    return output


def tf_inverse_warp(input, flow, device):
    '''
    linear interpolation for Image Warping
    '''
    shape = input.shape
    i_H = shape[1]
    i_W = shape[2]
    shape = flow.shape
    N = shape[0]
    H = shape[1]
    W = shape[2]

    N_i = torch.range(0, N-1)
    W_i = torch.range(0, W-1)
    H_i = torch.range(0, H-1)

    # n, h, w = torch.ndgrid(N_i, H_i, W_i)
    # n, h, w = torch.meshgrid(N_i, H_i, W_i, indexing='ij')
    n, h, w = meshgrid(N_i, H_i, W_i, indexing='ij')
    n = torch.from_numpy(n)
    h = torch.from_numpy(h)
    w = torch.from_numpy(w)
    # n = n.view(5,224,112,1)
    # h = h.view(5,224,112,1)
    # w = w.view(5,224,112,1)

    n = torch.unsqueeze(n, axis=3).to(device)
    h = torch.unsqueeze(h, axis=3).to(device)
    w = torch.unsqueeze(w, axis=3).to(device)

    # n = tf.cast(n, tf.float32)
    # h = tf.cast(h, tf.float32)
    # w = tf.cast(w, tf.float32)

    # v_col, v_row = torch.split(flow, 2, dim=-1)
    v_col = flow
    v_row = torch.zeros(flow.shape).to(device)
    # v_row = flow

    v_r0 = torch.floor(v_row)
    v_r1 = v_r0 + 1
    v_c0 = torch.floor(v_col)
    v_c1 = v_c0 + 1

    # H_ = tf.cast(i_H - 1, tf.float32)
    # W_ = tf.cast(i_W - 1, tf.float32)
    H_ = i_H - 1
    W_ = i_W - 1
    i_r0 = torch.clamp(h + v_r0, 0., H_)
    i_r1 = torch.clamp(h + v_r1, 0., H_)
    i_c0 = torch.clamp(w + v_c0, 0., W_)
    i_c1 = torch.clamp(w + v_c1, 0., W_)

    i_r0c0 = torch.cat((n, i_r0, i_c0), dim=-1)
    i_r0c1 = torch.cat((n, i_r0, i_c1), dim=-1)
    i_r1c0 = torch.cat((n, i_r1, i_c0), dim=-1)
    i_r1c1 = torch.cat((n, i_r1, i_c1), dim=-1)
    # x=input[0].view(1,224,224,3)
    # idx=i_r0c0[0].view(1,224,112,3)
    # idx=idx.type(torch.LongTensor)
    # aa=x[list((torch.arange(x.size(0)), *idx.chunk(2, 1)))]

    f00 = gather_nd(input.cpu(), i_r0c0.cpu())
    f01 = gather_nd(input.cpu(), i_r0c1.cpu())
    f10 = gather_nd(input.cpu(), i_r1c0.cpu())
    f11 = gather_nd(input.cpu(), i_r1c1.cpu())
    f00 = f00.to(device)
    f01 = f01.to(device)
    f10 = f10.to(device)
    f11 = f11.to(device)

    # f00 = f00.view(5,224,112,1)
    # f01 = f01.view(5,224,112,1)
    # f10 = f10.view(5,224,112,1)
    # f11 = f11.view(5,224,112,1)

    w00 = (v_r1 - v_row) * (v_c1 - v_col)
    w01 = (v_r1 - v_row) * (v_col - v_c0)
    w10 = (v_row - v_r0) * (v_c1 - v_col)
    w11 = (v_row - v_r0) * (v_col - v_c0)

    out = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    return out


class Model2(nn.Module):
    def __init__(self, device, base, incr_channel, incr_channel2, hsp1, hsp2, cls_m, cls, concat_r, concat_1, mask_branch, intra, demo_mode=False):
        super(Model2, self).__init__()
        self.base = nn.ModuleList(base)
        self.sp1 = hsp1
        self.sp2 = hsp2
        self.cls_m = cls_m
        self.cls = cls
        self.incr_channel1 = nn.ModuleList(incr_channel)
        self.incr_channel2 = nn.ModuleList(incr_channel2)
        self.concat4 = nn.ModuleList(concat_r)
        self.concat3 = nn.ModuleList(concat_r)
        self.concat2 = nn.ModuleList(concat_r)
        self.concat1 = nn.ModuleList(concat_1)
        self.mask = nn.ModuleList(mask_branch)
        self.extract = [13, 23, 33, 43]
        self.device = device
        self.group_size = 5
        self.intra = nn.ModuleList(intra)
        self.transformer_1 = Transformer(512, 4, 4, 782, group=self.group_size)
        self.transformer_2 = Transformer(512, 4, 4, 782, group=self.group_size)
        self.demo_mode = demo_mode

        self.conv1 = nn.Conv2d(1, 1, (224, 1), stride=1, padding=0)
        self.convTemp = nn.Conv2d(1, 1, 5, dilation=5, stride=1, padding=10)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(128, 1, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.conv2 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 512, 3, stride = 1, padding = 1)
        self.conv44 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        # self.conv4444 = nn.Conv2d(1024, 1024, 3, stride = 1, padding = 1)
        self.conv444 = nn.Conv2d(512, 128, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 3, 3, stride = 1, padding = 1)

        dim_in = 3
        dim_out = 256
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(3, affine=True, track_running_stats=True))
        
        self.modelAE = Network2()
        # self.conv6 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1)
        # self.conv7 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        # self.conv8 = nn.Conv2d(128, 256, 3, stride = 1, padding = 1)
        # self.conv88 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
        # self.conv888 = nn.Conv2d(256, 128, 3, stride = 1, padding = 1)
        # self.conv9 = nn.Conv2d(128, 3, 3, stride = 1, padding = 1)
        nc=3
        ngf=224
        ndf=224
        latent_variable_size=300
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*4, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*4)

        self.e5 = nn.Conv2d(ndf*4, ndf*4, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*4)

        self.fc1 = nn.Linear(43904, latent_variable_size)
        self.fc2 = nn.Linear(43904, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*4*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*4*2, ngf*4, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*4, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.TwoHeadsNetwork1 = TwoHeadsNetwork()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        # h5 = h5.view(-1, self.ndf*8*4*4) 

        h5 = h5.view(-1, 896* 7* 7) 

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # if args.cuda:
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #     eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*4*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(self.up5(h5)))))
        # return self.leakyrelu(self.d6(self.pd5(self.up5(self.up5(h5)))))
        # return self.d6(self.pd5(self.up5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward2(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

    def forward(self, x, attn):

        #     device = "cuda:0"

        # def model22(x, attn):

        device = "cuda:0"
        x_input = x

        # mu, log_var = self.encode(x_input)
        # z = self.reparameterize(mu, log_var)
        # ss=  [self.decode(z), x_input, mu, log_var]



        attn = attn.view(5, 1, 224, 224)
        attn = kornia.filters.gaussian_blur2d(attn, (11, 11), sigma=(1, 1))
        kernel = torch.ones(21, 21)
        attn = kornia.morphology.dilation(attn, kernel.to(device))
        attn = torch.where(attn < float(attn.mean()), 0.0, 255.0) / 255
        # attn = self.convTemp(attn)
        # attn = torch.where(attn < float(attn.mean()), 0.0, 255.0) / 255
        # attn = 1 - attn

        temp_out = warp(x, attn)

        mask_pred_mid = torch.zeros(5, 224, 112)
        x_input_mid = torch.zeros(5, 3, 224, 112)
        attn_input_mid = torch.zeros(5, 1, 224, 112)
        x_input_final = torch.zeros(5, 3, 224, 224)
        x_input_finalr = torch.zeros(5, 3, 224, 224)
        x_mid_final = torch.zeros(5, 3, 224, 112)
        x22 = torch.zeros(5, 3, 224, 112)

        for kk in range(x.shape[0]):
            # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
            x_input_mid[kk] = fn.resize(x[kk], size=[224, 112])
            attn_input_mid[kk] = fn.resize(attn[kk], size=[224, 112])

        x = x_input_mid.to(device)
        # cv2.imwrite("x_input_mid05.jpg", x_input_mid[2].permute(1,2,0).cpu().detach().numpy()*255)
        attn_input_mid2 = attn_input_mid.to(device)

        # attn_input_mid2 = attn_input_mid.to(self.device)+0.3 * attn_input_mid2
        # attn_input_mid2=torch.where(attn_input_mid2<float(attn_input_mid2.mean()), 0.0, 255.0)

        # cc = bb.tile((1,1,224,1))
        # cv2.imwrite("attn.jpg", attn[0].permute(1,2,0).cpu().detach().numpy()*255)
        # cv2.imwrite("attn2.jpg", cc[0].permute(1,2,0).cpu().detach().numpy()*255)

        # # cc=torch.where(cc<float(cc.mean()), 0.0, 255.0)
        # # cc = kornia.filters.box_blur(cc, (11,11), border_type='reflect', normalized=True)
        # cc = kornia.filters.gaussian_blur2d(cc, (11,11), sigma=(1,1))

        gray = (attn_input_mid2[0].permute(1, 2, 0).cpu(
        ).detach().numpy()*255).astype(np.uint8)*255
        img = gray
        thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]

        # thresh = cv2.threshold(gray, 4, 255, 0)[1]

        # apply morphology open to smooth the outline
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # find contours
        cntrs = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        # Contour filtering to get largest area
        area_thresh = 0
        for c in cntrs:
            area = cv2.contourArea(c)
            if area > area_thresh:
                area = area_thresh
                big_contour = c
        aa = []
        for i in range(len(cntrs)):
            aa = np.append(aa, cntrs[i])

        if len(aa) != 0:
            x_counters = []
            for jj in range(len(aa)):
                if jj % 2 == 0:
                    x_counters.append(int(aa[jj]))

            # draw the contour on a copy of the input image
            # results = img.copy()
            # #cv2.drawContours(results,[big_contour],0,(0,0,255),2)
            # x_counters = []
            # for count_er in big_contour:
            #     x=count_er[0][0]
            #     x_counters.append(x)

            # write result to disk
            # cv2.imwrite("greengreen_and_red_regions_threshold.png", thresh)
            # cv2.imwrite("green_and_red_regions_big_contour.png", results)

            # get contours
            # result = img.copy()
            # contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # contours = contours[0] if len(contours) == 2 else contours[1]

            # sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
            # largest_item= sorted_contours[0]

            # big_contour
            i_non_zero_start = min(x_counters)
            i_non_zero_end = max(x_counters)
            # save resulting image
            # cv2.imwrite('two_blobs_result.jpg',result*255)

            # noz = torch.nonzero(bb[0,0,0])

            # i_non_zero_start = float(noz[0])
            # i_non_zero_end = float(noz[-1])

            v0 = []
            for i in range(int(i_non_zero_start)):
                v0.append(i)

            v1 = []
            for i in range(int(i_non_zero_start), int(i_non_zero_end)):
                v1.append(i)

            v2 = []
            for i in range(int(i_non_zero_end), int(112)):
                v2.append(i)
            # cc = kornia.filters.box_blur(cc, (11,11), border_type='reflect', normalized=True)

            v0_F = False
            v1_F = False
            v2_F = False
            v0_s = 0
            v1_s = 0
            v2_s = 0

            if len(v0) > 10:
                v0_F = True
                v0_s = 1
                x0 = torch.zeros(5, 3, 224, len(v0)-int(10)).to(device)
                x0r = torch.zeros(5, 3, 224, len(v0)).to(device)
                for kk in range(x.shape[0]):
                    # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
                    x0[kk] = fn.resize(x_input_mid[kk, :, :, 0:len(v0)], size=[
                                       224, len(v0)-int(10)])

            if len(v1) > 10:
                v1_F = True
                v1_s = 1
                x1 = torch.zeros(5, 3, 224, len(v1)+int(20)).to(device)
                x1r = torch.zeros(5, 3, 224, len(v1)).to(device)
                for kk in range(x.shape[0]):
                    # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
                    x1[kk] = fn.resize(x_input_mid[kk, :, :, len(
                        v0)+1:len(v0)+1+len(v1)], size=[224, len(v1)+int(20)])

            if len(v2) > 10:
                v2_F = True
                v2_s = 1
                x2 = torch.zeros(5, 3, 224, len(v2)-int(10)).to(device)
                x2r = torch.zeros(5, 3, 224, len(v2)).to(device)
                for kk in range(x.shape[0]):
                    # mask_pred_mid[kk] = fn.resize(mask_pred[kk].view(1,224,224), size=[112,224])
                    x2[kk] = fn.resize(x_input_mid[kk, :, :, len(
                        v0)+1+len(v1):len(v0)+1+len(v1)+len(v2)], size=[224, len(v2)-int(10)])
            if v0_F and v1_F and v2_F:
                ff = torch.cat((x0, x1, x2), dim=-1)

                for kk in range(x.shape[0]):
                    x0r[kk] = fn.resize(x0[kk, :, :, :], size=[224, len(v0)])

                for kk in range(x.shape[0]):
                    x1r[kk] = fn.resize(x1[kk, :, :, :], size=[224, len(v1)])
                for kk in range(x.shape[0]):
                    x2r[kk] = fn.resize(x2[kk, :, :, :], size=[224, len(v2)])

                ffr = torch.cat((x0r, x1r, x2r), dim=-1)

                # cv2.imwrite("retarg.jpg", ff[2].permute(1,2,0).cpu().detach().numpy()*255)
                # cv2.imwrite("retargBack.jpg", ffr[2].permute(1,2,0).cpu().detach().numpy()*255)
            else:
                if v1_F and v2_F:
                    ff = torch.cat((x1, x2), dim=-1)

                    for kk in range(x.shape[0]):
                        x1r[kk] = fn.resize(
                            x1[kk, :, :, :], size=[224, len(v1)])
                    for kk in range(x.shape[0]):
                        x2r[kk] = fn.resize(
                            x2[kk, :, :, :], size=[224, len(v2)])

                    ffr = torch.cat((x1r, x2r), dim=-1)

                else:
                    if v0_F and v1_F:
                        ff = torch.cat((x0, x1), dim=-1)
                        for kk in range(x.shape[0]):
                            x0r[kk] = fn.resize(
                                x0[kk, :, :, :], size=[224, len(v0)])
                        for kk in range(x.shape[0]):
                            x1r[kk] = fn.resize(
                                x1[kk, :, :, :], size=[224, len(v1)])

                        ffr = torch.cat((x0r, x1r), dim=-1)

                    if v0_F and v2_F:
                        ff = torch.cat((x0, x2), dim=-1)
                        for kk in range(x.shape[0]):
                            x0r[kk] = fn.resize(
                                x0[kk, :, :, :], size=[224, len(v0)])
                        for kk in range(x.shape[0]):
                            x2r[kk] = fn.resize(
                                x2[kk, :, :, :], size=[224, len(v2)])

                        ffr = torch.cat((x0r, x2r), dim=-1)

                    if v1_F and not v2_F and not v0_F:
                        ff = x1
                        for kk in range(x.shape[0]):
                            x1r[kk] = fn.resize(
                                x1[kk, :, :, :], size=[224, len(v1)])

                        ffr = x1r

            if v0_s + v1_s + v2_s < 2:
                ff = x
                ffr = x

            aaa = ff
            aaar = ffr

            for kk in range(x.shape[0]):
                x_input_final[kk] = fn.resize(aaa[kk], size=[224, 224])
                x_input_finalr[kk] = fn.resize(aaar[kk], size=[224, 224])

            # cv2.imwrite("aaa05ObjectBig.jpg", aaa[2].permute(1,2,0).cpu().detach().numpy()*255)

            x_input_final = x_input_final.to(device)
            x_input_finalr = x_input_finalr.to(device)
            x_input_finalT = torch.zeros(5, 3, 224, 112)

            for kk in range(x.shape[0]):
                x_input_finalT[kk] = fn.resize(
                    x_input_final[kk], size=[224, 112])

            cv2.imwrite("resizedNormal.jpg", (x_input_finalT[2].permute(
                1, 2, 0).cpu().detach().numpy())*255)


            # x23 = self.forward2(x_input_final.clone())
            # x23 = x23[0]
            # x2 = torch.zeros(5, 3, 224, 224)
            # for kk in range(x23.shape[0]):
            #     x2[kk] = fn.resize(x23[kk], size=[224, 224])
            # x2=x2.to(self.device)
            # Do the reverse

            x2mid = F.relu(self.conv2(x_input_final.clone()), inplace=False)
            x2mid = F.relu(self.conv3(x2mid), inplace=False)
            x2mid = F.relu(self.conv4(x2mid), inplace=False)
            x2mid = F.relu(self.conv44(x2mid), inplace=False)
            # x2mid = F.relu(self.conv4444(x2mid), inplace=False)
            x2mid = F.relu(self.conv444(x2mid), inplace=False)
            x2mid = F.relu(self.conv5(x2mid), inplace=False)
            
            x2mid=self.TwoHeadsNetwork1(x2mid)
            x2mid = x2mid[1].clone()
            # x2 = x2/x2.max()

            x2mid  = x2mid.clone() + x_input_final 

            # for kk in range(x.shape[0]):
            #     x22[kk] = fn.resize(x2[kk], size=[224,112])
            # cv2.imwrite("pred_imgResized.jpg", (x22[2].permute(1,2,0).cpu().detach().numpy())*255)
            # cv2.imwrite("pred_imgResized.jpg", 255*(x22[2] / (x22[2].max().clone() - x22[2].min().clone())).permute(1,2,0).cpu().detach().numpy())

    # x2 reverse

            i_non_zero_start = min(x_counters)
            i_non_zero_end = max(x_counters)
            # save resulting image
            # cv2.imwrite('two_blobs_result.jpg',result*255)

            # noz = torch.nonzero(bb[0,0,0])

            # i_non_zero_start = float(noz[0])
            # i_non_zero_end = float(noz[-1])

            v0 = []
            for i in range(int(i_non_zero_start)):
                v0.append(i)

            v1 = []
            for i in range(int(i_non_zero_start), int(i_non_zero_end)):
                v1.append(i)

            v2 = []
            for i in range(int(i_non_zero_end), int(112)):
                v2.append(i)
            # cc = kornia.filters.box_blur(cc, (11,11), border_type='reflect', normalized=True)

            v0_F = False
            v1_F = False
            v2_F = False
            v0_s = 0
            v1_s = 0
            v2_s = 0

            if len(v0) > 10:
                v0_F = True
                v0_s = 1
                x0r = torch.zeros(5, 3, 224, len(v0))

            if len(v1) > 10:
                v1_F = True
                v1_s = 1
                x1r = torch.zeros(5, 3, 224, len(v1))

            if len(v2) > 10:
                v2_F = True
                v2_s = 1
                x2r = torch.zeros(5, 3, 224, len(v2))
            if v0_F and v1_F and v2_F:
                ff = torch.cat((x0, x1, x2), dim=-1)

                for kk in range(x.shape[0]):
                    x0r[kk] = fn.resize(x0[kk, :, :, :], size=[224, len(v0)])

                for kk in range(x.shape[0]):
                    x1r[kk] = fn.resize(x1[kk, :, :, :], size=[224, len(v1)])
                for kk in range(x.shape[0]):
                    x2r[kk] = fn.resize(x2[kk, :, :, :], size=[224, len(v2)])

                ffr = torch.cat((x0r, x1r, x2r), dim=-1)

                # cv2.imwrite("retarg.jpg", ff[2].permute(1,2,0).cpu().detach().numpy()*255)
                # cv2.imwrite("retargBack.jpg", ffr[2].permute(1,2,0).cpu().detach().numpy()*255)
            else:
                if v1_F and v2_F:
                    ff = torch.cat((x1, x2), dim=-1)

                    for kk in range(x.shape[0]):
                        x1r[kk] = fn.resize(
                            x1[kk, :, :, :], size=[224, len(v1)])
                    for kk in range(x.shape[0]):
                        x2r[kk] = fn.resize(
                            x2[kk, :, :, :], size=[224, len(v2)])

                    ffr = torch.cat((x1r, x2r), dim=-1)

                else:
                    if v0_F and v1_F:
                        ff = torch.cat((x0, x1), dim=-1)
                        for kk in range(x.shape[0]):
                            x0r[kk] = fn.resize(
                                x0[kk, :, :, :], size=[224, len(v0)])
                        for kk in range(x.shape[0]):
                            x1r[kk] = fn.resize(
                                x1[kk, :, :, :], size=[224, len(v1)])

                        ffr = torch.cat((x0r, x1r), dim=-1)

                    if v0_F and v2_F:
                        ff = torch.cat((x0, x2), dim=-1)
                        for kk in range(x.shape[0]):
                            x0r[kk] = fn.resize(
                                x0[kk, :, :, :], size=[224, len(v0)])
                        for kk in range(x.shape[0]):
                            x2r[kk] = fn.resize(
                                x2[kk, :, :, :], size=[224, len(v2)])

                        ffr = torch.cat((x0r, x2r), dim=-1)

                    if v1_F and not v2_F and not v0_F:
                        ff = x1
                        for kk in range(x.shape[0]):
                            x1r[kk] = fn.resize(
                                x1[kk, :, :, :], size=[224, len(v1)])

                        ffr = x1r

            if v0_s + v1_s + v2_s < 2:
                ff = x
                ffr = x

            aaa = ff
            aaar = ffr

            for kk in range(x.shape[0]):
                x_input_final[kk] = fn.resize(aaa[kk], size=[224, 224])
                x_input_finalr[kk] = fn.resize(aaar[kk], size=[224, 224])
        # else:
        #     aaa = torch.zeros(5,3,224,112)
        #     aaa2 = torch.zeros(5,3,224,112)

    ####################################
            # x3 = self.modelAE(x_input_finalr)
            # x3 = x_input_finalr + self.main(x_input_finalr)
            x33, mu, logvar = self.forward2(x_input_finalr.clone())
            
            x3 = torch.zeros(5, 3, 224, 224)
            for kk in range(x33.shape[0]):
                x3[kk] = fn.resize(x33[kk], size=[224, 224])
            
            x3=x3.to(self.device)
            x3 = x3+x_input_finalr
            # x3 = F.relu(self.conv6(x_input_finalr.clone()), inplace=False)
            # x3 = F.relu(self.conv7(x3), inplace=False)
            # x3 = F.relu(self.conv8(x3), inplace=False)
            # x3 = F.relu(self.conv88(x3), inplace=False)
            # x3 = F.relu(self.conv888(x3), inplace=False)
            # x3 = F.relu(self.conv9(x3), inplace=False)

        else:
            x33, mu, logvar = self.forward2(x_input_finalr.to(device))
            x2mid = x_input
            x3 = x_input
            aaa = torch.zeros(5, 3, 224, 112)
            # aaa2 = torch.zeros(5,3,224,112)
        # return x_input_final.to(device), aaa, x_input_finalr.to(device)
        return x2mid, aaa, x3, mu, logvar, temp_out
        # return x_input_final.to(self.device), aaa


# build the whole network
def build_model(device, demo_mode=False):
    return Model2(device,
                  vgg(base['vgg']),
                  incr_channel(),
                  incr_channel2(),
                  hsp(512, 64),
                  hsp(64**2, 32),
                  cls_modulation_branch(32**2, 512),
                  cls_branch(512, 78),
                  concat_r(),
                  concat_1(),
                  mask_branch(),
                  intra(), demo_mode)


def build_model2(device, demo_mode=False):
    return Model2(device,
                  vgg(base['vgg']),
                  incr_channel(),
                  incr_channel2(),
                  hsp(512, 64),
                  hsp(64**2, 32),
                  cls_modulation_branch(32**2, 512),
                  cls_branch(512, 78),
                  concat_r(),
                  concat_1(),
                  mask_branch(),
                  intra(), demo_mode)

# weight init


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
