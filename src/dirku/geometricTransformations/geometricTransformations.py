import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor
from ..interpolation import *
from ..numericalIntegration import *

class nonrigidDeformation():
    """Class for deformable transformation, based on velocity fields.
            :param pts: points in material configuration
            :type pts: torch.Tensor
            :param integrator: numerical integration method to solve teh initial value problem for the connecting differential equations
            :type integrator:  trapezoidal or forwardEuler class
            :param interpolator: interpolation method for interpolating velocity fields
            :type interpolator: interpolation class
    """
    def __init__(self,pts: Tensor,integrator: Union[trapezoidal,forwardEuler],interpolator: Union[nearest,linear,cubic]):
        """Constructor method."""
        self.pts=pts
        self.integrator = integrator
        self.interpolator = interpolator
    def __call__(self,velocityField: Tensor,mainTerms: Optional[list] = None,regTerms: Optional[list] = None)-> Tuple[Tensor,Tensor]:
        """Displaces points with nonrigid displacement. Evaluates mainTerm and regTerms on the fly.
        :param velocityField: set of velocity fields (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.Tensor
        :param mainTerms: main terms of the cost function, for example similarity measure or collision detection
        :type mainTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :param regTerms: regularization terms of the cost function
        :type regTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :return ptsDis: displaced points in spatial configuration
        :rtype ptsDis: torch.Tensor
        :return loss: accumulated loss of mainTerm and regTerms
        :rtype loss: torch.Tensor
        """
        ptsDis,loss=self.integrator(self.pts,velocityField,self.interpolator,mainTerms,regTerms)
        return ptsDis,loss
    def apply(self,pts: Tensor,velocityField: Tensor)->Tensor:
        """Displaces points with nonrigid displacement.
        :param velocityField: set of velocity fields (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.Tensor
        :return ptsDis: displaced points as cartesian coordinates (# points, dim)
        :rtype ptsDis: torch.Tensor
        """
        ptsDis, loss = self.integrator(pts, velocityField, self.interpolator, None, None)
        return ptsDis

class affineTransformation():
    """ Class for affine displacement.
    :param pts: points in material configuration
    :type pts: torch.Tensor
    """
    def __init__(self,pts: Tensor):
        """Constructor method."""
        self.pts=pts
        self.dimension = pts.size(1)
        self.device = pts.device
    def __call__(self, affineMat: Tensor,mainTerms: Optional[list] = None,regTerms: Optional[list] = None)->Tuple[Tensor,Tensor]:
        """ Displaces points with an affine transformation. Evaluates mainTerm and regTerms on the fly.
        :param affineMat: affine transformation matrix; either 3x3 or 4x4;
        :type affineMat: torch.Tensor
        :param mainTerms: main terms of the cost function, for example similarity measure or collision detection
        :type mainTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :param regTerms: regularization terms of the cost function
        :type regTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :return ptsDis: displaced points in spatial configuration
        :rtype ptsDis: torch.Tensor
        :return loss: accumulated loss of mainTerms and regTerms
        :rtype loss: torch.Tensor
        """
        ptsDis = torch.cat((self.pts, torch.ones((self.pts.shape[0], 1), device=self.device, dtype=self.pts.dtype)), 1)
        ptsDis = affineMat.mm(ptsDis.t()).t()[:, :self.dimension]
        loss=0
        if mainTerms is not None:
            for term in mainTerms:
                loss=loss+term(dis=ptsDis-self.pts)
        elif regTerms is not None:
            for term in regTerms:
                loss=loss+term(dis=ptsDis-self.pts)
        else:
            loss=None
        return ptsDis,loss
    def apply(self,pts: Tensor,affineMat: Tensor)-> Tensor:
        """ Displaces points with an affine transformation.
        :param affineMat: affine transformation matrix; either 3x3 or 4x4;
        :type affineMat: torch.Tensor
        :return ptsDis: displaced points in spatial configuration
        :rtype ptsDis: torch.Tensor
        """
        ptsDis = torch.cat((pts, torch.ones((pts.shape[0], 1), device=self.device, dtype=pts.dtype)), 1)
        ptsDis = affineMat.mm(ptsDis.t()).t()[:, :self.dimension]
        return ptsDis




