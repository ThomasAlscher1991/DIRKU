import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor

class linear2d:
    """ Class for bi-linear interpolation in 2 dimensions.
    :param device: computation device, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel or voxel
    :type scale: torch.Tensor
    :param jac: switch to compute jacobian
    :type jac: bool
    :param lap: switch to compute laplacian
    :type lap: bool
    """
    def __init__(self,device: str,scale: Tensor,jac: bool,lap: bool):
        """Constructor method."""
        self.device=device
        self.scale=scale
        self.ones4 = torch.ones([2, 2], dtype=torch.int32, device=device)
        rangi = torch.arange(0, 2, device=device).int()
        self.stride_x, self.stride_y = torch.meshgrid(
            [rangi, rangi])
        self.jac=jac
        self.lap=lap
    def __call__(self,pts: Tensor,data: Tensor)->Tuple[Tensor,Tensor,Tensor]:
        """ Compute tri-linear approximation in 2 dimensions.
        :param pts: interpolation points coordinates
        :type pts: torch.Tensor
        :param data: tensor of data points (# fields,  dim 1,  dim 2 )
        :type data: torch.Tensor
        :return w: values at interpolation points (# fields,#points)
        :rtype w: torch.Tensor
        :return jacobian: jacobian of interpolation values
        :rtype jacobian: torch.Tensor
        :return laplacian: laplacian of interpolation values
        :rtype laplacian: torch.Tensor
        """
        p = torch.arange(pts.size(0))
        t_idx = pts.div(self.scale).floor()
        t = pts.div(self.scale) - t_idx
        onesp = torch.ones(t.size(0), dtype=torch.int32, device=self.device)
        t_idx = t_idx.flatten()
        indices = torch.clamp(
            torch.einsum('a,bc->abc', t_idx[2 * p], self.ones4) + torch.einsum('a,bc->abc', onesp, self.stride_x.int()), 0,
            data.shape[1] - 1) * (data.size(2))
        indices = indices + torch.clamp(
            torch.einsum('a,bc->abc', t_idx[2 * p + 1], self.ones4) + torch.einsum('a,bc->abc', onesp, self.stride_y.int()),
            0,
            data.shape[2] - 1)
        tf = t.flatten()
        y = torch.stack([1 - tf, tf], dim=1)
        w = torch.sum(
            torch.einsum('ab,ac->abc', y[2 * p, :], y[2 *p + 1, :]) * data.flatten(start_dim=1)[:, indices.long()],
            dim=[2, 3])
        if self.jac:
            dy = torch.stack([t.flatten().clone() * 0 - 1, t.flatten().clone() * 0 + 1], dim=1)
            wx = torch.sum(
                torch.einsum('ab,ac->abc', dy[2 * p, :], y[2 * p + 1, :]) * data.flatten(start_dim=1)[:, indices.long()],
                dim=[2, 3])
            wy = torch.sum(
                torch.einsum('ab,ac->abc', y[2 * p, :], dy[2 * p + 1, :]) * data.flatten(start_dim=1)[:, indices.long()],
                dim=[2, 3])
            jacobian = torch.stack([wx.t(), wy.t()], dim=2)
        else:
            jacobian=0
        laplacian=torch.zeros((1, pts.size(0)),device=self.device)
        return w,jacobian,laplacian





class linear3d:
    """ Class for tri-linear interpolation in 3 dimensions.
    :param device: computation device, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel or voxel
    :type scale: torch.Tensor
    :param jac: switch to compute jacobian
    :type jac: bool
    :param lap: switch to compute laplacian
    :type lap: bool
    """
    def __init__(self,device: str,scale: Tensor,jac: bool,lap: bool):
        """Constructor method."""
        self.device=device
        self.scale = scale
        self.ones4 = torch.ones([2, 2, 2], dtype=torch.int32, device=device)
        self.stride_x, self.stride_y, self.stride_z = torch.meshgrid(
            [torch.arange(0, 2, device=device), torch.arange(0, 2, device=device), torch.arange(0, 2, device=device)])
        self.jac=jac
        self.lap=lap
    def __call__(self,pts: Tensor,data: Tensor)->Tuple[Tensor,Tensor,Tensor]:
        """ Compute tri-linear approximation in 3 dimensions.
        :param pts: interpolation points coordinates
        :type pts: torch.Tensor
        :param data: tensor of data points (# fields,  dim 1,  dim 2, dim 3 )
        :type data: torch.Tensor
        :return w: values at interpolation points (# fields,#points)
        :rtype w: torch.Tensor
        :return jacobian: jacobian of interpolation values
        :rtype jacobian: torch.Tensor
        :return laplacian: laplacian of interpolation values
        :rtype laplacian: torch.Tensor
        """
        t_idx = pts.mul(1 / self.scale).floor()
        t = pts.mul(1 / self.scale) - t_idx
        p = torch.arange(pts.size(0))
        onesp = torch.ones(t.size(0), dtype=torch.int32, device=self.device)
        t_idx = t_idx.flatten()
        indices = torch.clamp(
            torch.einsum('a,bcd->abcd', t_idx[3 * p], self.ones4) + torch.einsum('a,bcd->abcd', onesp, self.stride_x.int()),
            0,
            data.shape[1] - 1) * (data.size(2) * data.size(3))
        indices = indices + torch.clamp(
            torch.einsum('a,bcd->abcd', t_idx[3 * p + 1], self.ones4) + torch.einsum('a,bcd->abcd', onesp,
                                                                                     self.stride_y.int()),
            0,
            data.shape[2] - 1) * (data.size(3))
        indices = indices + torch.clamp(
            torch.einsum('a,bcd->abcd', t_idx[3 * p + 2], self.ones4) + torch.einsum('a,bcd->abcd', onesp,
                                                                                     self.stride_z.int()),
            0,
            data.shape[3] - 1)
        y = torch.stack([1 - t.flatten(), t.flatten()], dim=1)
        w = torch.sum(
            torch.einsum('ab,ac,ad->abcd', y[3 * p, :], y[3 * p + 1, :], y[3 * p + 2, :]) * data.flatten(start_dim=1)[:,
                                                                                            indices.long()],dim=[2, 3, 4])
        if self.jac:
            dy = torch.stack([t.flatten().clone() * 0 - 1, t.flatten().clone() * 0 + 1], dim=1)
            wx = torch.sum(
                torch.einsum('ab,ac,ad->abcd', dy[3 * p, :], y[3 * p + 1, :], y[3 * p + 2, :]) * data.flatten(start_dim=1)[
                                                                                                 :, indices.long()],
                dim=[2, 3, 4])
            wy = torch.sum(
                torch.einsum('ab,ac,ad->abcd', y[3 * p, :], dy[3 * p + 1, :], y[3 * p + 2, :]) * data.flatten(start_dim=1)[
                                                                                                 :, indices.long()],
                dim=[2, 3, 4])
            wz = torch.sum(
                torch.einsum('ab,ac,ad->abcd', y[3 * p, :], y[3 * p + 1, :], dy[3 * p + 2, :]) * data.flatten(start_dim=1)[
                                                                                                 :, indices.long()],
                dim=[2, 3, 4])
            jacobian = torch.stack([wx.t(), wy.t(), wz.t()], dim=2)
        else:
            jacobian=0
        laplacian=torch.zeros((1, pts.size(0)),device=self.device)
        return w,jacobian,laplacian



class linear:
    """ Class for linear interpolation.
    :param device: computation device, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel or voxel
    :type scale: torch.Tensor
    :param jac: switch to compute jacobian
    :type jac: bool
    :param lap: switch to compute laplacian
    :type lap: bool
    """
    def __new__(cls, device: str, scale: Tensor,jac: Optional[bool] = False,lap: Optional[bool] = False)->Union[linear2d,linear3d]:
        """Static method. Decides on dimensionality.
        :return: instance of interpolation class
        :rtype: interpolation class"""
        if scale.size(0) == 2:
            return linear2d(device, scale,jac,lap)
        elif scale.size(0) == 3:
            return linear3d(device, scale,jac,lap)
        else:
            raise ValueError("Unsupported dimension. Only 2D and 3D are supported.")

