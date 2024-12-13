import torch


class intersectionDetection():
    """ Class for collision detection based on signed distance fields.
    :param pts: points used for collision detection
    :type pts: torch.tensor
    :param sdf: signed distance map of scene with objects that should be avoided
    :type sdf: torch.tensor
    :param interpolator: CUDA device or cpu, see torch docs
    :type interpolator: interpolation class
    :param coef: coefficient applied to the collision loss
    :type coef: float
    """
    def __init__(self, pts,sdf,interpolator, coef=1.,pointsMask=None,pointsMaskLabel=None):
        """constructor method"""
        self.sdf = sdf
        self.interpolator=interpolator
        self.coef=coef
        self.pts=pts
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
    def __call__(self,dis=0,**kwargs):
        """ Calculates the summed depth of intersecting points. Adds tiny to prevent exploding gradients.
        :param dis: displacement of pts
        :type dis: torch.tensor
        :return: summed depth of intersecting points
        :rtype: torch.tensor
        """
        if self.pointsMask is not None:
            dis=dis[self.pointsMask==self.pointsMaskLabel]
            pts=self.pts[self.pointsMask==self.pointsMaskLabel]
            sdf_int, _, _ = self.interpolator(pts + dis, self.sdf)
        else:
            sdf_int,_,_=self.interpolator(self.pts+dis, self.sdf)
        loss = torch.sqrt((sdf_int** 2)+ torch.finfo().tiny)

        return torch.sum(loss)*self.coef/dis.size(0)
        #return torch.sum(loss)*self.coef


