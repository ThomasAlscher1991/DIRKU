import torch


class linOp():
    """ Class for linear operator regularization of velocity fields
    :param coef: weight of linOp
    :type coef: float
    :param lam: coefficient for laplacian term
    :type lam: float
    :param mu: coefficient for interpolation term
    :type mu: float
    """
    def __init__(self,coef=1,mu=1,lam=1,pointsMask=None,pointsMaskLabel=None):
        """constructor method"""
        self.mu=mu
        self.lam=lam
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel

    def __call__(self,vel=0,vel_lap=0,**kwargs):
        """Calculates the linear operator.
            :param vel: points velocities
            :type vel: torch.tensor
            :param vel_lap: laplacian of point velocities
            :type vel_lap: torch.tensor
            :return: linear operator norm
            :rtype: torch.tensor"""
        if self.pointsMask is not None:
            vel = vel[self.pointsMask == self.pointsMaskLabel]
            vel_lap = vel_lap[self.pointsMask == self.pointsMaskLabel]
        else:
            pass
        return self.coef*torch.sum(torch.norm(self.mu*vel_lap+self.lam*vel.t(),dim=1))


