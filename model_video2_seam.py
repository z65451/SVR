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
from Intra_MLP import index_points, knn_l2
import torchvision.transforms.functional as fn
import itertools
import cv2
from torch.nn import functional as f

from torch.autograd import Variable

from AE import Network as Network2

from skimage import io, transform, util
from skimage import filters, color

from vformer.vformer.models.classification import ViViTModel3



import sys
sys.path.append('NonUniformBlurKernelEstimation/')
sys.path.append('motionComp/')
sys.path.append('seamcarving/')

# from motionComp.ran2 import motionCompensation
from seamcarving.api.carver import crop
from seamc2 import  cropByColumn
from NonUniformBlurKernelEstimation.models.TwoHeadsNetwork import TwoHeadsNetwork

x_ratio = 224
# x_ratio = 360
# x_ratio = 320
# x_ratio = 672

def seam_carve(img, f, n, s):
    """
    Helper function to recalculate the energy map after each seam removal
    
    :param img: image to be carved
    :param f: energy map function
    :param n: number of seams to remove
    """
    s=s + np.random.normal(5, 15.9, s.shape)
    s =255* (s - s.min()) / (s.max() - s.min())
    for ii in range(3):
        img[:,:,ii] = img[:,:,ii] + s/5
    for i in range(n):
        eimg = cv2.resize(s, (img.shape[1], img.shape[0])) 
        # eimg = filters.sobel(color.rgb2gray(eimg))
        # eimg = f(img)
        img = transform.seam_carve(img, eimg, 'vertical', 1)
    return img, eimg

def slow_dual_gradient(img):
    height = img.shape[0]
    width = img.shape[1]
    energy = np.empty((height, width))
    for i in range(height):
        for j in range(width):
            L = img[i, (j-1) % width]
            R = img[i, (j+1) % width]
            U = img[(i-1) % height, j]
            D = img[(i+1) % height, j]
            
            dx_sq = np.sum((R - L)**2)
            dy_sq = np.sum((D - U)**2)
            energy[i,j] = np.sqrt(dx_sq + dy_sq)
    return energy
def slow_forward_energy(img):
    height = img.shape[0]
    width = img.shape[1]
    
    I = color.rgb2gray(img)
    energy = np.zeros((height, width))
    m = np.zeros((height, width))
    
    for i in range(1, height):
        for j in range(width):
            up = (i-1) % height
            down = (i+1) % height
            left = (j-1) % width
            right = (j+1) % width
    
            mU = m[up,j]
            mL = m[up,left]
            mR = m[up,right]
                
            cU = np.abs(I[i,right] - I[i,left])
            cL = np.abs(I[up,j] - I[i,left]) + cU
            cR = np.abs(I[up,j] - I[i,right]) + cU
            
            cULR = np.array([cU, cL, cR])
            mULR = np.array([mU, mL, mR]) + cULR
            
            argmin = np.argmin(mULR)
            m[i,j] = mULR[argmin]
            energy[i,j] = cULR[argmin]
            
    return energy

def forward_energy(img, flag=False):
    height = img.shape[0]
    width = img.shape[1]
    I = color.rgb2gray(img)
    
    energy = np.zeros((height, width))
    m = np.zeros((height, width))
    
    U = np.roll(I, 1, axis=0)
    L = np.roll(I, 1, axis=1)
    R = np.roll(I, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, height):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
        
    return energy

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
    # device = "cpu"
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

    n = torch.unsqueeze(n, axis=3).to(device)
    h = torch.unsqueeze(h, axis=3).to(device)
    w = torch.unsqueeze(w, axis=3).to(device)

    # n = tf.cast(n, tf.float32)
    # h = tf.cast(h, tf.float32)
    # w = tf.cast(w, tf.float32)

    # v_col, v_row = torch.split(flow, 2, dim=-1)
    v_col = flow[:,:,:,0]
    v_row = flow[:,:,:,1]
    v_col = torch.unsqueeze(v_col, axis=3).to(device)
    v_row = torch.unsqueeze(v_row, axis=3).to(device)
    # v_col = flow
    # v_row = torch.zeros(flow.shape).to(device)
    # v_row = flow

    v_r0 = torch.floor(v_row)
    v_r1 = v_r0 + 1
    v_c0 = torch.floor(v_col)
    v_c1 = v_c0 + 1

    H_ = i_H - 1
    W_ = i_W - 1
    # H_ = -i_H + 1
    # W_ = -i_W + 1
    i_r0 = torch.clamp(h + v_r0, 0., H_)
    i_r1 = torch.clamp(h + v_r1, 0., H_)
    i_c0 = torch.clamp(w + v_c0, 0., W_)
    i_c1 = torch.clamp(w + v_c1, 0., W_)

    i_r0c0 = torch.cat((n, i_r0, i_c0), dim=-1)
    i_r0c1 = torch.cat((n, i_r0, i_c1), dim=-1)
    i_r1c0 = torch.cat((n, i_r1, i_c0), dim=-1)
    i_r1c1 = torch.cat((n, i_r1, i_c1), dim=-1)
    
    i_r0c0.type=torch.int32
    i_r0c1.type=torch.int32
    i_r1c0.type=torch.int32
    i_r1c1.type=torch.int32

    
    f00 = torch_gather_nd5(input.cpu(), i_r0c0.cpu().long())
    f01 = torch_gather_nd5(input.cpu(), i_r0c1.cpu().long())
    f10 = torch_gather_nd5(input.cpu(), i_r1c0.cpu().long())
    f11 = torch_gather_nd5(input.cpu(), i_r1c1.cpu().long())
    # f00 = torch_gather_nd5(input, i_r0c0.long())
    # f01 = torch_gather_nd5(input, i_r0c1.long())
    # f10 = torch_gather_nd5(input, i_r1c0.long())
    # f11 = torch_gather_nd5(input, i_r1c1.long())
    f00 = f00.to(device)
    f01 = f01.to(device)
    f10 = f10.to(device)
    f11 = f11.to(device)
    
    # f00 = torch.unsqueeze(f00, axis=3).to(device)
    # f01 = torch.unsqueeze(f01, axis=3).to(device)
    # f10 = torch.unsqueeze(f10, axis=3).to(device)
    # f11 = torch.unsqueeze(f11, axis=3).to(device)

    w00 = (v_r1 - v_row) * (v_c1 - v_col)
    w01 = (v_r1 - v_row) * (v_col - v_c0)
    w10 = (v_row - v_r0) * (v_c1 - v_col)
    w11 = (v_row - v_r0) * (v_col - v_c0)

    out = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    return out


class Flow_spynet(nn.Module):
    def __init__(self, spynet_pretrained=None):
        super(Flow_spynet, self).__init__()
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def forward(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward' is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lrs.size()    
        # assert h >= 64 and w >= 64, ('The height and width of inputs should be at least 64, 'f'but got {h} and {w}.')
        
        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        lrs_1 = torch.cat([lrs[:, 0, :, :, :].unsqueeze(1), lrs], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        lrs_2 = torch.cat([lrs, lrs[:, t-1, :, :, :].unsqueeze(1)], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t+1, 2, h, w)         # [b, 6, 2, 64, 64]
        flows_backward = flows_backward[:, 1:, :, :, :]                          # [b, 5, 2, 64, 64]

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t+1, 2, h, w)      # [b, 6, 2, 64, 64]
            flows_forward = flows_forward[:, :-1, :, :, :]                       # [b, 5, 2, 64, 64]

        return flows_forward, flows_backward

class Model2(nn.Module):
    def __init__(self, device, base, incr_channel, incr_channel2, hsp1, hsp2, cls_m, cls, concat_r, concat_1, mask_branch, intra, demo_mode=False):
        super(Model2, self).__init__()
        self.base = nn.ModuleList(base)
        self.sp1 = hsp1
        self.sp2 = hsp2
        self.cls_m = cls_m
        self.cls = cls
        self.device = device
        # self.incr_channel1 = nn.ModuleList(incr_channel)
        # self.incr_channel2 = nn.ModuleList(incr_channel2)
        # self.concat4 = nn.ModuleList(concat_r)
        # self.concat3 = nn.ModuleList(concat_r)
        # self.concat2 = nn.ModuleList(concat_r)
        # self.concat1 = nn.ModuleList(concat_1)
        # self.mask = nn.ModuleList(mask_branch)
        # self.extract = [13, 23, 33, 43]
        # self.device = device
        # self.group_size = 5
        # self.intra = nn.ModuleList(intra)
        # self.transformer_1 = Transformer(512, 4, 4, 782, group=self.group_size)
        # self.transformer_2 = Transformer(512, 4, 4, 782, group=self.group_size)
        self.demo_mode = demo_mode

        self.conv1 = nn.Conv2d(1, 1, (224, 1), stride=1, padding='valid')
        # self.convTemp = nn.Conv2d(1, 1, 5, dilation=5, stride=1, padding=10)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(128, 1, 3, stride = 1, padding = 1)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.conv2 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.conv44 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        # self.conv4444 = nn.Conv2d(1024, 1024, 3, stride = 1, padding = 1)
        self.conv444 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 3, 3, stride = 1, padding = 1)

        dim_in = 3
        dim_out = 256
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ourViViTModel4 = ViViTModel3(
                img_size=224,
                patch_t = 5,
                patch_h=16,
                patch_w=16,
                in_channels=3,
                n_classes=10,
                num_frames=5,
                embedding_dim=196,
                depth=12,
                num_heads=3,
                head_dim=6,
                p_dropout=0.2)
        self.init_feature = nn.Conv2d(3, 8, 3, 1, 1, bias=True)

        self.deep_feature = RDG(G0=8, C=4, G=24, n_RDB=4)
        self.pamLike = PAM(8)
        self.init_feature2 = nn.Conv2d(8, 3, 3, 1, 1, bias=True)

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

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward2(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

    def forward(self, x, attn, x_r, attn_r, disp_l, disp_r):

        device = "cuda:0"
        x_input = x
        x_input_r = x_r
        x_left = torch.unsqueeze(x,dim=0)
        x_right = torch.unsqueeze(x_r,dim=0)
        
        disp_l = torch.unsqueeze(disp_l,dim=0)
        disp_r = torch.unsqueeze(disp_r,dim=0)

        transOut_l = self.ourViViTModel4(x_left, disp_l)
        transOut_r = self.ourViViTModel4(x_right, disp_r)

        attn_border = attn.view(5,1,224,448)
        attn_border = kornia.filters.gaussian_blur2d(attn_border, (11,11), sigma=(1,1))
        attn_border=torch.where(attn_border<float(attn_border.mean()), 0.0, 255.0)
        
        # R
        attn_border_r = attn_r.view(5,1,224,448)
        attn_border_r = kornia.filters.gaussian_blur2d(attn_border_r, (11,11), sigma=(1,1))
        attn_border_r=torch.where(attn_border_r<float(attn_border_r.mean()), 0.0, 255.0)

        attn=attn.unsqueeze(0)
        attn = attn.permute(1,0,2,3)

        # R
        attn_r=attn_r.unsqueeze(0)
        attn_r = attn_r.permute(1,0,2,3)

        # attn = attn.view(5, 1, 224, 224)
        attn = kornia.filters.gaussian_blur2d(attn, (11, 11), sigma=(1, 1))
        kernel = torch.ones(11, 11)
        attn = kornia.morphology.dilation(attn, kernel.to(device))
        attn_r = kornia.morphology.dilation(attn_r, kernel.to(device))
        for ir in range(5):
            attn[ir] = 255 * (((attn[ir].clone()-attn[ir].min())  / (attn[ir].max().clone() - attn[ir].min().clone())))
            attn_r[ir] = 255 * (((attn_r[ir].clone()-attn_r[ir].min())  / (attn_r[ir].max().clone() - attn_r[ir].min().clone())))

        cv2.imwrite("attn1_l.jpg", attn[2,0].cpu().detach().numpy())
        cv2.imwrite("attn1_r.jpg", attn_r[2,0].cpu().detach().numpy())

        x_input_final_seam = torch.zeros(5, 3, 224, x_ratio)
        x_input_final_seam_r = torch.zeros(5, 3, 224, x_ratio)
        
        temp_out = 1

        x_input_mid = torch.zeros(5, 3, 224, x_ratio)
        attn_input_mid = torch.zeros(5, 1, 224, x_ratio)
        attn_input_border = torch.zeros(5, 1, 224, x_ratio)
        x_input_final = torch.zeros(5, 3, 224, 448)
        
        x_input_mid_r = torch.zeros(5, 3, 224, x_ratio)
        attn_input_mid_r = torch.zeros(5, 1, 224, x_ratio)
        attn_input_border_r = torch.zeros(5, 1, 224, x_ratio)
        x_input_final_r = torch.zeros(5, 3, 224, 448)


        x_input_finalr = torch.zeros(5, 3, 224, 448)
        x_input_finalr_r = torch.zeros(5, 3, 224, 448)
        for kk in range(x.shape[0]):
            x_input_mid[kk] = fn.resize(x[kk], size=[224, x_ratio])
            attn_input_mid[kk] = fn.resize(attn[kk], size=[224, x_ratio])
            attn_input_border[kk] = fn.resize(attn_border[kk], size=[224, x_ratio])
            
            x_input_mid_r[kk] = fn.resize(x_r[kk], size=[224, x_ratio])
            attn_input_mid_r[kk] = fn.resize(attn_r[kk], size=[224, x_ratio])
            attn_input_border_r[kk] = fn.resize(attn_border_r[kk], size=[224, x_ratio])
        attn_input_border = attn_input_border.to(device)
        attn_input_border_r = attn_input_border_r.to(device)
        x = x_input_mid.to(device)
        x_r = x_input_mid_r.to(device)

        a1 =  attn_input_mid.to(device)
        a1 = a1.to(device)
        a2 = self.conv1(a1)
        for i_5 in range(5):
            for i_sum in range(x_ratio):
                a2[i_5, :,:,i_sum] = torch.sum(a1[i_5, :, :, i_sum])
            
        a2 = a2.permute(0,1,3,2)
        a2 = a2.tile((1,224,1,1))
        a2_temp = torch.zeros(5, 224, x_ratio, 1)

        for ia2 in range(5):
            a2_temp[ia2] = 255 * (((a2[ia2].clone()-a2[ia2].min().clone())  / (a2[ia2].max().clone() - a2[ia2].min().clone())))

        # R
        a1_r =  attn_input_mid_r.to(device)
        a1_r = a1_r.to(device)
        a2_r = self.conv1(a1_r)
        for i_5 in range(5):
            for i_sum in range(x_ratio):
                a2_r[i_5, :,:,i_sum] = torch.sum(a1_r[i_5, :, :, i_sum])
            
        a2_r = a2_r.permute(0,1,3,2)
        a2_r = a2_r.tile((1,224,1,1))
        a2_temp_r = torch.zeros(5, 224, x_ratio, 1)

        for ia2 in range(5):
            a2_temp_r[ia2] = 255 * (((a2_r[ia2].clone()-a2_r[ia2].min().clone())  / (a2_r[ia2].max().clone() - a2_r[ia2].min().clone())))


        a2 = 255 - a2_temp.to(device)
        cc = 255 * (((a2[2].clone()-a2[2].min().clone())  / (a2[2].max().clone() - a2[2].min().clone()))).cpu().detach().numpy()
        cv2.imwrite("a2.jpg", cc)
        a =  1.9 * (a2) + 255-a1.permute(0,2,3, 1).clone()
        normed =  torch.cumsum(a, dim=2) / (torch.sum(a, 2, keepdim=True))
        a = normed * (448-x_ratio)#+attn_input_mid.permute(0,2,3,1).to(device)/2
        cc = 255 * (((a[2].clone()-a[2].min().clone())  / (a[2].max().clone() - a[2].min().clone())))

        cv2.imwrite("a.jpg", a[2].cpu().detach().numpy())
        flow = torch.cat([a, torch.zeros(a.shape).to(self.device)], -1)

        x_input2 = x_input.unsqueeze(dim=2)
        b, c, t, h, w = x_input2.shape
        x_input2  =x_input2.permute(2,1,0,3,4)

        # R
        a2_r = 255 - a2_temp_r.to(device)
        cc_r = 255 * (((a2_r[2].clone()-a2_r[2].min().clone())  / (a2_r[2].max().clone() - a2_r[2].min().clone()))).cpu().detach().numpy()
        cv2.imwrite("a2.jpg", cc_r)
        a_r =  1.9 * (a2_r) + 255-a1_r.permute(0,2,3, 1).clone()
        normed_r =  torch.cumsum(a_r, dim=2) / (torch.sum(a_r, 2, keepdim=True))
        a_r = normed_r * (448-x_ratio)#+attn_input_mid.permute(0,2,3,1).to(device)/2
        cc_r = 255 * (((a_r[2].clone()-a_r[2].min().clone())  / (a_r[2].max().clone() - a_r[2].min().clone())))
        cv2.imwrite("a.jpg", a_r[2].cpu().detach().numpy())
        flow_r = torch.cat([a_r, torch.zeros(a_r.shape).to(self.device)], -1)

        img = tf_inverse_warp(x_input.permute(0,2,3,1), flow, device)
        img[int(img.shape[0]/2)] = img[int(img.shape[0]/2)].to(device)
        img_l_out = img
        
        x_input_final_seam = img.permute(0,3,1,2).to(device)
        transOut2 = fn.resize(transOut_l[0], size=[224, x_ratio])
        temp = 255 * (((img[2].clone()-img[2].min().clone())  / (img[2].max().clone() - img[2].min().clone())))
        cv2.imwrite("out1_l.jpg", temp.cpu().detach().numpy())
        x_input_final_seam[2] = x_input_final_seam[2] + transOut2.to(device)

        # R

        img_r = tf_inverse_warp(x_input_r.permute(0,2,3,1), flow_r, device)
        img_r[int(img_r.shape[0]/2)] = img_r[int(img_r.shape[0]/2)].to(device)
        
        img_r_out = img_r


        x_input_final_seam_r = img_r.permute(0,3,1,2).to(device)
        transOut2_r = fn.resize(transOut_r[0], size=[224, x_ratio])
        temp_r = 255 * (((img_r[2].clone()-img_r[2].min().clone())  / (img_r[2].max().clone() - img_r[2].min().clone())))
        cv2.imwrite("out1_r.jpg", temp_r.cpu().detach().numpy())
        x_input_final_seam_r[2] = x_input_final_seam_r[2] + transOut2_r.to(device)
        
        # PAM
        buffer_left = self.relu(self.init_feature(x_input_final_seam))
        buffer_right = self.relu(self.init_feature(x_input_final_seam_r))

        buffer_left, catfea_left = self.deep_feature(buffer_left)
        buffer_right, catfea_right = self.deep_feature(buffer_right)

        buffer_leftT, buffer_rightT, M_right_to_left, M_left_to_right, V_left, V_right = self.pamLike(buffer_left, buffer_right, catfea_left, catfea_right, 1)


        x_input_final_seam = self.relu(self.init_feature2(buffer_leftT))
        x_input_final_seam_r = self.relu(self.init_feature2(buffer_rightT))


        i_non_zero_start = []
        i_non_zero_end = []
        for i_border in range(5):
            gray = (attn_input_border[i_border].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)*255
            img = gray 

            thresh = cv2.threshold(gray,4,255,0)[1]

            # apply morphology open to smooth the outline
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # find contours
            cntrs = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            # Contour filtering to get largest area
            area_thresh = 0
            for c in cntrs:
                area = cv2.contourArea(c)
                if area > area_thresh:
                    area = area_thresh
                    big_contour = c
            aa=[]
            for i in range(len(cntrs)):
                aa=np.append(aa,cntrs[i])
            
            if len(aa ) != 0:
                x_counters = []
                for jj in range(len(aa)):
                    if jj%2==0:
                        x_counters.append(int(aa[jj]))

                i_non_zero_start.append(min(x_counters))
                i_non_zero_end.append(max(x_counters))
            else:
                i_non_zero_start.append(0)
                i_non_zero_end.append(x_ratio)
            

        x_input_final_seam = x_input_final_seam.to(device)

        x2mid = F.relu(self.conv2(x_input_final_seam.clone()), inplace=False)
        x2mid = F.relu(self.conv3(x2mid), inplace=False)
        x2mid = F.relu(self.conv4(x2mid), inplace=False)
        x2mid = F.relu(self.conv44(x2mid), inplace=False)
        x2mid = F.relu(self.conv444(x2mid), inplace=False)
        x2mid = F.relu(self.conv5(x2mid), inplace=False)

        x_input_final = x_input_final.to(device)
        x_input_finalr = x_input_finalr.to(device)
        # x_input_finalT = torch.zeros(5, 3, 224, x_ratio)
        x_input_finalT2 = torch.zeros(5, 3, 224, 448)
        x2mid  = x2mid.clone() + x_input_final_seam
        for kk in range(x.shape[0]):
            x_input_finalT2[kk] = fn.resize(
                x2mid[kk], size=[224, 448])
        x2mid = x_input_finalT2
        x2mid = x2mid.to(device)
        x3 = x_input_finalT2.to(device).clone()


        x3 = x3+x_input_finalr


        i_non_zero_start_r = []
        i_non_zero_end_r = []
        for i_border in range(5):
            gray_r = (attn_input_border_r[i_border].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)*255
            img_r = gray_r 


            thresh = cv2.threshold(gray_r,4,255,0)[1]

            # apply morphology open to smooth the outline
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # find contours
            cntrs = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

            # Contour filtering to get largest area
            area_thresh = 0
            for c in cntrs:
                area = cv2.contourArea(c)
                if area > area_thresh:
                    area = area_thresh
                    big_contour = c
            aa=[]
            for i in range(len(cntrs)):
                aa=np.append(aa,cntrs[i])
            
            if len(aa ) != 0:
                x_counters = []
                for jj in range(len(aa)):
                    if jj%2==0:
                        x_counters.append(int(aa[jj]))

                i_non_zero_start_r.append(min(x_counters))
                i_non_zero_end_r.append(max(x_counters))
            else:
                i_non_zero_start_r.append(0)
                i_non_zero_end_r.append(x_ratio)
            

        x_input_final_seam_r = x_input_final_seam_r.to(device)

        # for iMOC in range(4):
        #     x_input_final_seam[iMOC] = motionCompensation(x_input_final_seam[iMOC], x_input_final_seam[iMOC+1])
        x2mid = F.relu(self.conv2(x_input_final_seam_r.clone()), inplace=False)
        x2mid = F.relu(self.conv3(x2mid), inplace=False)
        x2mid = F.relu(self.conv4(x2mid), inplace=False)
        x2mid = F.relu(self.conv44(x2mid), inplace=False)
        x2mid = F.relu(self.conv444(x2mid), inplace=False)
        x2mid = F.relu(self.conv5(x2mid), inplace=False)


        x_input_final_r = x_input_final_r.to(device)
        x_input_finalr_r = x_input_finalr_r.to(device)
        # x_input_finalT_r = torch.zeros(5, 3, 224, x_ratio)
        x_input_finalT2_r = torch.zeros(5, 3, 224, 448)
        x2mid_r  = x2mid.clone() + x_input_final_seam_r
        for kk in range(x.shape[0]):
            x_input_finalT2_r[kk] = fn.resize(
                x2mid_r[kk], size=[224, 448])
        x2mid_r = x_input_finalT2_r
        x2mid_r = x2mid_r.to(device)
        x3_r = x_input_finalT2_r.to(device).clone()


        x3_r = x3_r+x_input_finalr_r
        
        return x2mid, a1, x3, temp_out, i_non_zero_start, i_non_zero_end, x2mid_r, a1_r, x3_r, temp_out, i_non_zero_start_r, i_non_zero_end_r, img_l_out, img_r_out, M_right_to_left, M_left_to_right, V_left, V_right


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

class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)
    
class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x
class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out, buffer_cat


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class ResB3(nn.Module):
    def __init__(self, channels):
        super(ResB3, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.bq = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.bs = nn.Conv2d(4*channels, channels, 1, 1, 0, groups=4, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(4 * channels)
        self.bn = nn.BatchNorm2d(4 * channels)

    def __call__(self, x_left, x_right, catfea_left, catfea_right, is_training):
        b, c0, h0, w0 = x_left.shape
        Q = self.bq(self.rb(self.bn(catfea_left)))
        b, c, h, w = Q.shape
        Q = Q - torch.mean(Q, 3).unsqueeze(3).repeat(1, 1, 1, w)
        K = self.bs(self.rb(self.bn(catfea_right)))
        K = K - torch.mean(K, 3).unsqueeze(3).repeat(1, 1, 1, w)

        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),                    # (B*H) * Wl * C
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))                    # (B*H) * C * Wr
        M_right_to_left = self.softmax(score)                                                   # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(score.permute(0, 2, 1))                                  # (B*H) * Wr * Wl

        M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        V_left_tanh = torch.tanh(5 * V_left)
        V_right_tanh = torch.tanh(5 * V_right)

        x_leftT = torch.bmm(M_right_to_left, x_right.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        x_rightT = torch.bmm(M_left_to_right, x_left.permute(0, 2, 3, 1).contiguous().view(-1, w0, c0)
                            ).contiguous().view(b, h0, w0, c0).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        out_left = x_left * (1 - V_left_tanh.repeat(1, c0, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c0, 1, 1)
        out_right = x_right * (1 - V_right_tanh.repeat(1, c0, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c0, 1, 1)

        if is_training == 1:
            return out_left, out_right, M_right_to_left, M_left_to_right, V_left, V_right
        if is_training == 0:
            return out_left, out_right, M_right_to_left, M_left_to_right, V_left, V_right
        

def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed