import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor

class svk():
    """ Class for Saint Venant–Kirchhoff regularization.
    :param coef: weight of SVK
    :type coef: float
    :param lam: lamé constant
    :type lam: float
    :param mu: lamé constant
    :type mu: float
    :param pointsMask: a mask for pts if only a subset of pts needs to be measured
    :type pointsMask: torch.Tensor
    :param pointsMaskLabel: the mask label for pointsMask that needs to be measured
    :type pointsMaskLabel: int
    """
    def __init__(self,coef: float=1,lam: float=1,mu: float=1,pointsMask: Optional[Tensor] = None,pointsMaskLabel: Optional[Tensor] = None):
        """Constructor method."""
        self.lam=lam
        self.mu=mu
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel

    def __call__(self,dis_jac: Tensor=0,**kwargs)->Tensor:
        """Calculates SVK.
            :param dis_jac: jacobians of point displacements
            :type dis_jac: torch.Tensor
            :return: accumulated summed strainEnergy at time step
            :rtype: torch.Tensor"""
        #id = torch.eye(self.pts.size(1),device=self.pts.device).repeat((self.pts.size(0), 1, 1))
        #f=dis_jac+id
        if self.pointsMask is not None:
            dis_jac = dis_jac[self.pointsMask == self.pointsMaskLabel]
        else:
            pass
        jac = dis_jac
        jact = torch.transpose(jac, dim0=1, dim1=2)
        GreenTensor = 0.5 * (jac + jact + torch.einsum('abc,ade->abe', jact, jac))
        strainEnergy = (self.lam / 2 * torch.einsum('abb', GreenTensor) ** 2 + self.mu * torch.sum(GreenTensor ** 2,
                                                                                                   dim=[1, 2]))
        return self.coef*torch.sum(strainEnergy)
