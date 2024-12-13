import torch

class linear:
    """ Class for selecting dimension appropriate linear approximation.
    :param device: CUDA device or cpu, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel fx torch.tensor([1.,1.]) for 2D
    :type scale: torch.tensor
    :return: interpolation class
    :rtype: interpolation class
    """
    def __new__(cls, device, scale,jac=False,lap=False):
        """Static method. Decides on dimensionality.
        :return: instance of interpolation class
        :rtype: interpolation class"""
        if scale.size(0) == 2:
            return linear2d(device, scale,jac,lap)
        elif scale.size(0) == 3:
            return linear3d(device, scale,jac,lap)
        else:
            raise ValueError("Unsupported dimension. Only 2D and 3D are supported.")




class linear2d:
    """ Class for bi-linear approximation in 2 dimensions.
    :param device: CUDA device or cpu, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel
    :type scale: torch.tensor
    """
    def __init__(self,device,scale,jac,lap):
        """Constructor method
                """
        self.device=device
        self.scale=scale
        self.ones4 = torch.ones([2, 2], dtype=torch.int32, device=device)
        rangi = torch.arange(0, 2, device=device).int()
        self.stride_x, self.stride_y = torch.meshgrid(
            [rangi, rangi])
        self.jac=jac
        self.lap=lap
    def __call__(self,pts,data):
        """ Compute bi-linear approximation in 2 dimensions.
        :param pts: interpolation points as tensor (#points,dim) of cartesian coordinates
        :type pts: torch.tensor
        :param x: tensor of data points (# fields,  dim 1,  dim 2 )
        :type x: torch.tensor
        :return: values at interpolation points (# fields,#points)
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
    """ Class for tri-linear approximation in 3 dimensions.
    :param device: CUDA device or cpu, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel
    :type scale: torch.tensor
    """
    def __init__(self,device,scale,jac,lap):

        """Constructor method
                """
        self.device=device
        self.scale = scale
        self.ones4 = torch.ones([2, 2, 2], dtype=torch.int32, device=device)
        self.stride_x, self.stride_y, self.stride_z = torch.meshgrid(
            [torch.arange(0, 2, device=device), torch.arange(0, 2, device=device), torch.arange(0, 2, device=device)])
        self.jac=jac
        self.lap=lap
    def __call__(self,pts,data):
        """ Compute tri-linear approximation in 3 dimensions.
        :param pts: interpolation points as tensor (#points,dim) of cartesian coordinates
        :param x: tensor of interpolation data (# fields,  dim 1,  dim 2 , dim 3)
        :return: values at interpolation points (# fields,#points)
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





