import torch





class svk():
    """ Class for Saint Venant–Kirchhoff regularization.
    :param coef: weight of SVK
    :type coef: float
    :param lam: lamé constant
    :type lam: float
    :param mu: lamé constant
    :type mu: float
    """
    def __init__(self,coef=1,lam=1,mu=1,pointsMask=None,pointsMaskLabel=None):
        """constructor method"""
        self.lam=lam
        self.mu=mu
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel

    def __call__(self,dis_jac=0,**kwargs):
        """Calculates SVK.
            :param dis_jac: jacobians of point displacements
            :type dis_jac: torch.tensor
            :return: accumulated summed strainEnergy at time step
            :rtype: torch.tensor"""
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
