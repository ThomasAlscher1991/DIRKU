import torch



class nearest:
    """ Class for nearest neighbour interpolation in d dimensions.
    :param device: CUDA device or cpu, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel
    :type scale: torch.tensor
    """
    def __new__(cls, device, scale):
        """Static method. Decides on dimensionality.
        :return: instance of interpolation class
        :rtype: interpolation class"""

        if scale.size(0) == 2:
            return nearest2D(device, scale)
        elif scale.size(0) == 3:
            return nearest3D(device, scale)
        else:
            raise ValueError("Unsupported dimension. Only 2D and 3D are supported.")

class nearest2D:
    """ Class for nearest neighbour interpolation in 2 dimensions.
    :param device: CUDA device or cpu, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel
    :type scale: torch.tensor
    """
    def __init__(self,device,scale=None):
        """Constructor method
                """
        self.device=device
        self.scale=scale
    def __call__(self,pts,x):
        """ Compute nearest neighbour interpolation in 3 dimensions.
        :param pts: interpolation points as tensor (#points,dim) of cartesian coordinates
        :type pts: torch.tensor
        :param x: tensor of data points (# fields,  (dim 1,  dim 2,...) )
        :type x: torch.tensor
        :return: values at interpolation points (# fields,#points)
        """
        if self.scale is None:
            self.scale=torch.ones(pts.size(1),device=self.device)
        t_idx = torch.round(pts.div(self.scale)).long()
        w=torch.zeros((x.size(0),pts.size(0))).to(device=self.device)
        t_idx[:, 0] = torch.clamp(t_idx[:, 0], 0, x.size(1)-1)
        t_idx[:, 1] = torch.clamp(t_idx[:, 1], 0, x.size(2)-1)
        for i,field in enumerate(x):
            w[i]=field[t_idx[:,0],t_idx[:,1]]
        return w


class nearest3D:
    """ Class for nearest neighbour interpolation in 3 dimensions.
    :param device: CUDA device or cpu, see torch docs
    :type device: str
    :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel
    :type scale: torch.tensor
    """
    def __init__(self,device,scale=None):
        """Constructor method
                """
        self.device=device
        self.scale=scale
    def __call__(self,pts,x):
        """ Compute nearest neighbour interpolation in 3 dimensions.
        :param pts: interpolation points as tensor (#points,dim) of cartesian coordinates
        :type pts: torch.tensor
        :param x: tensor of data points (# fields,  (dim 1,  dim 2,...) )
        :type x: torch.tensor
        :return: values at interpolation points (# fields,#points)
        """
        if self.scale is None:
            self.scale=torch.ones(pts.size(1),device=self.device)
        t_idx = torch.round(pts.div(self.scale)).long()
        w=torch.zeros((x.size(0),pts.size(0))).to(device=self.device)
        t_idx[:, 0] = torch.clamp(t_idx[:, 0], 0, x.size(1)-1)
        t_idx[:, 1] = torch.clamp(t_idx[:, 1], 0, x.size(2)-1)
        t_idx[:, 2] = torch.clamp(t_idx[:, 2], 0, x.size(3)-1)
        for i,field in enumerate(x):
            w[i]=field[t_idx[:,0],t_idx[:,1],t_idx[:,2]]
        return w