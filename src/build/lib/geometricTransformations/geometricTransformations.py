import torch
#from src.dirku import utils,interpolation, geometricTransformations,similarityMeasure, optimization, numericalIntegration, regularization

class nonrigidDeformation():
    """Class for nonrigid displacement, based on velocity fields.
            :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
            :type pts: torch.tensor
            :param integrator: numerical integration method to solve differential equations
            :type integrator:numericalIntegration class
            :param interpolator: interpolation method for interpolating velocity fields
            :type interpolator: interpolation class
    """
    def __init__(self,pts,integrator,interpolator,pointsMask=None):
        """Constructor method"""
        self.pts=pts
        self.integrator = integrator
        self.interpolator = interpolator
        self.pointsMask=pointsMask
    def __call__(self,velocityField,mainTerm=None,regTerms=None):
        """Displaces points with nonrigid displacement. Evaluates mainTerm and regTerms on the fly. Used in optimization.
        :param velocityField: set of velocity fields (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :param mainTerm: main term of the cost function, for example similarity measure or collision detection
        :type mainTerm: class
        :param regTerms: regularization term of the cost function, for example l2 or linear operator regularizer
        :type regTerms: class
        :return ptsDis: displaced points as cartesian coordinates (# points, dim)
        :rtype ptsDis: torch.tensor
        :return loss: accumulated loss of mainTerm and regTerms
        :rtype loss: torch.tensor
        """
        ptsDis,loss=self.integrator(self.pts,velocityField,self.interpolator,mainTerm,regTerms)
        return ptsDis,loss
    def apply(self,pts,velocityField):
        """Displaces points with nonrigid displacement. Used in postprocessing.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param velocityField: set of velocity fields (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :return ptsDis: displaced points as cartesian coordinates (# points, dim)
        :rtype ptsDis: torch.tensor
        """
        ptsDis, loss = self.integrator(pts, velocityField, self.interpolator, None, None)
        return ptsDis

class affineTransformation():
    """ Class for affine displacement.
    :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
    :type pts: torch.tensor
    :return: displaced points as cartesian coordinates (# points, dim)
    :rtype: torch.tensor
    """
    def __init__(self,pts):
        """constructor method"""
        self.pts=pts
        self.dimension = pts.size(1)
        self.device = pts.device
    def __call__(self, affineMat,mainTerm=None,regTerms=None):
        """ Displaces points with an affine transformation.
        :param affineMat: affine transformation matrix; either 3x3 or 4x4;
        :type affineMat: torch.tensor
        :return ptsDis: displaced points as cartesian coordinates (# points, dim)
        :rtype ptsDis: torch.tensor
        :return loss: accumulated loss of mainTerm and regTerms
        :rtype loss: torch.tensor
        """
        ptsDis = torch.cat((self.pts, torch.ones((self.pts.shape[0], 1), device=self.device, dtype=self.pts.dtype)), 1)
        ptsDis = affineMat.mm(ptsDis.t()).t()[:, :self.dimension]
        if mainTerm is not None:
            loss=0
            loss=loss+mainTerm(dis=ptsDis-self.pts)
            for term in regTerms:
                loss=loss+term(dis=ptsDis-self.pts)
        else:
            loss=None
        return ptsDis,loss
    def apply(self,pts,x):
        """ Displaces points with an affine transformation.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param affineMat: affine transformation matrix; either 3x3 or 4x4;
        :type affineMat: torch.tensor
        :return : displaced points as cartesian coordinates (# points, dim)
        :rtype : torch.tensor
        """
        tpst = torch.cat((pts, torch.ones((pts.shape[0], 1), device=self.device, dtype=pts.dtype)), 1)
        tpts = x.mm(tpst.t()).t()[:, :self.dimension]
        return tpts




