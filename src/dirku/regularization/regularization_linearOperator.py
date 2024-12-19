import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor

class linOp():
    """ Class for linear operator regularization of velocity fields
    :param coef: weight of linOp
    :type coef: float
    :param lam: coefficient for laplacian term
    :type lam: float
    :param mu: coefficient for identity term
    :type mu: float
    :param pointsMask: a mask for pts if only a subset of pts needs to be regularized
    :type pointsMask: torch.Tensor
    :param pointsMaskLabel: the mask label for pointsMask that needs to be regularized
    :type pointsMaskLabel: int
    """
    def __init__(self,coef: float=1,mu: float=1,lam: float=1,pointsMask: Optional[Tensor] = None,pointsMaskLabel: Optional[Tensor] = None):
        """constructor method"""
        self.mu=mu
        self.lam=lam
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel

    def __call__(self,vel: Tensor=0,vel_lap: Tensor=0,**kwargs)->Tensor:
        """Calculates the linear operator.
            :param vel: points velocities
            :type vel: torch.Tensor
            :param vel_lap: laplacian of point velocities
            :type vel_lap: torch.Tensor
            :return: linear operator norm
            :rtype: torch.Tensor"""
        if self.pointsMask is not None:
            vel = vel[self.pointsMask == self.pointsMaskLabel]
            vel_lap = vel_lap[self.pointsMask == self.pointsMaskLabel]
        else:
            pass
        return self.coef*torch.sum(torch.norm(self.mu*vel_lap+self.lam*vel.t(),dim=1))


