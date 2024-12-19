import torch
from typing import Optional, Type, Union
from torch import Tensor
from ..interpolation import *

class intersectionDetection:
    """ Class for collision detection based on signed distance fields (SDF). The deformable object is represented as a point cloud, the non-deformable obstacle as an SDF.
    :param pts: points representing the deformable object
    :type pts: torch.Tensor
    :param sdf: SDF of non-deformable obstacles
    :type sdf: torch.Tensor
    :param interpolator: interpolator for pts locations after displacement in SDF
    :type interpolator: nearest, linear, or cubic interpolation class
    :param coef: coefficient applied to the collision loss
    :type coef: float
    :param pointsMask: a mask for pts if only a subset of pts needs to be checked for collision
    :type pointsMask: torch.Tensor
    :param pointsMaskLabel: the mask label for pointsMask that needs to be checked for collision
    :type pointsMaskLabel: int
    """
    def __init__(self, pts: Tensor,sdf: Tensor,interpolator: Union[nearest,linear,cubic], coef: float=1.,pointsMask: Optional[Tensor] = None,pointsMaskLabel: Optional[int] = None):
        """Constructor method."""
        self.sdf = sdf
        self.interpolator=interpolator
        self.coef=coef
        self.pts=pts
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
    def __call__(self,dis: Tensor=None,**kwargs)->Tensor:
        """ Calculates the summed depth of intersecting points. Adds tiny to prevent exploding gradients.
        If pointsMask is given, only masked pts are checked.
        :param dis: displacement of pts
        :type dis: torch.Tensor
        :return: summed depth of intersecting points
        :rtype: torch.Tensor
        """
        if self.pointsMask is not None:
            dis=dis[self.pointsMask==self.pointsMaskLabel]
            pts=self.pts[self.pointsMask==self.pointsMaskLabel]
            sdf_int, _, _ = self.interpolator(pts + dis, self.sdf)
        else:
            sdf_int,_,_=self.interpolator(self.pts+dis, self.sdf)
        loss = torch.sqrt((sdf_int** 2)+ torch.finfo().tiny)
        return torch.sum(loss)*self.coef


