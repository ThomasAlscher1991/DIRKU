import torch

class l2():
    """ Class for L2 regularization of velocity fields.
    :param coef: weight of l2 norm
    :type coef: float
    """
    def __init__(self,coef=1,pointsMask=None,pointsMaskLabel=None):
        """constructor method"""
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel


    def __call__(self,vel=0,**kwargs):
        """Calculates L2 norm.
            :param vel: point velocities
            :type vel: torch.tensor
            :return: L2 norm
            :rtype: torch.tensor"""
        if self.pointsMask is not None:
            vel = vel[self.pointsMask == self.pointsMaskLabel]
        else:
            pass
        return self.coef*torch.sum(torch.norm(vel,dim=1))



