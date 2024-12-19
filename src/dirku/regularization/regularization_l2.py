import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor
class l2():
    """ Class for L2 regularization of velocity fields.
    :param coef: weight of l2 norm
    :type coef: float
    :param pointsMask: a mask for pts if only a subset of pts needs to be regularized
    :type pointsMask: torch.Tensor
    :param pointsMaskLabel: the mask label for pointsMask that needs to be regularized
    :type pointsMaskLabel: int
    """
    def __init__(self,coef: float=1,pointsMask: Optional[Tensor] = None,pointsMaskLabel: Optional[Tensor] = None):
        """Constructor method."""
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
    def __call__(self,vel: Tensor=0,**kwargs)->Tensor:
        """Calculates L2 norm.
            :param vel: point velocities
            :type vel: torch.Tensor
            :return: L2 norm
            :rtype: torch.tensor"""
        if self.pointsMask is not None:
            vel = vel[self.pointsMask == self.pointsMaskLabel]
        else:
            pass
        return self.coef*torch.sum(torch.norm(vel,dim=1))



