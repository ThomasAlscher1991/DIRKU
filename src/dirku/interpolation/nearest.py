import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor

class nearest2D:
    """ Class for nearest neighbour interpolation in 2 dimensions.
    :param device: computation device, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel or voxel
    :type scale: torch.Tensor
    """
    def __init__(self,device,scale=None):
        """Constructor method.                """
        self.device=device
        self.scale=scale
    def __call__(self,pts,data):
        """ Compute nearest neighbour interpolation in 2 dimensions. Can interpolate on one field of data points or stacked fields
        :param pts: interpolation points coordinates
        :type pts: torch.Tensor
        :param data: tensor of data points (# fields,  dim 1,  dim 2 )
        :type data: torch.Tensor
        :return w: values at interpolation points (# fields,#points)
        :rtype w: torch.Tensor
        """
        if self.scale is None:
            self.scale=torch.ones(pts.size(1),device=self.device)
        t_idx = torch.round(pts.div(self.scale)).long()
        w=torch.zeros((data.size(0),pts.size(0))).to(device=self.device)
        t_idx[:, 0] = torch.clamp(t_idx[:, 0], 0, data.size(1)-1)
        t_idx[:, 1] = torch.clamp(t_idx[:, 1], 0, data.size(2)-1)
        for i,field in enumerate(data):
            w[i]=field[t_idx[:,0],t_idx[:,1]]
        return w


class nearest3D:
    """ Class for nearest neighbour interpolation in 3 dimensions.
    :param device: computation device, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel or voxel
    :type scale: torch.Tensor
    """
    def __init__(self,device: str,scale: Tensor)->Tensor:
        """Constructor method."""
        self.device=device
        self.scale=scale
    def __call__(self,pts,data):
        """ Compute nearest neighbour interpolation in 3 dimensions. Can interpolate on one field of data points or stacked fields
        :param pts: interpolation points coordinates
        :type pts: torch.Tensor
        :param data: tensor of data points (# fields,  dim 1,  dim 2, dim 3 )
        :type data: torch.Tensor
        :return w: values at interpolation points (# fields,#points)
        :rtype w: torch.Tensor
        """
        t_idx = torch.round(pts.div(self.scale)).long()
        w=torch.zeros((data.size(0),pts.size(0))).to(device=self.device)
        t_idx[:, 0] = torch.clamp(t_idx[:, 0], 0, data.size(1)-1)
        t_idx[:, 1] = torch.clamp(t_idx[:, 1], 0, data.size(2)-1)
        t_idx[:, 2] = torch.clamp(t_idx[:, 2], 0, data.size(3)-1)
        for i,field in enumerate(data):
            w[i]=field[t_idx[:,0],t_idx[:,1],t_idx[:,2]]
        return w

class nearest:
    """ Class for nearest neighbour interpolation.
    :param device: computation device, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel or voxel
    :type scale: torch.Tensor
    """
    def __new__(cls, device: str, scale: Tensor)->Union[nearest2D,nearest3D]:
        """Static method. Decides on dimensionality.
        :return: instance of interpolation class
        :rtype: nearest2D or nearest3D class"""
        if scale.size(0) == 2:
            return nearest2D(device, scale)
        elif scale.size(0) == 3:
            return nearest3D(device, scale)
        else:
            raise ValueError("Unsupported dimension. Only 2D and 3D are supported.")