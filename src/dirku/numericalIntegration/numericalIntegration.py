import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor
from ..interpolation import *

class forwardEuler:
    """ Class for forward euler numerical integration of velocity fields. Time interval is assumed to be [0;1] and step size is assumed to be equidistant.
    :param t_steps: steps taken to cover time interval
    :type t_steps: int
    :param stationary: flag for stationary velocity fields
    :type stationary: bool
    """
    def __init__(self,t_steps: int,stationary: Optional[bool] = False):
        """ Constructor method.        """
        self.t_steps=t_steps
        self.stationary=stationary

    def __call__(self,pts: Tensor,velocityField: Tensor,interpolator: Union[nearest,linear,cubic],mainTerms: list,regTerms: list)->Tuple[Tensor,Tensor]:
        """ Numerical integration. Decides whether to treat the velocity field as stationary or non-stationary.
        :param pts: points coordinates in material configuration
        :type pts: torch.Tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.Tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: nearest, linear, or cubic interpolation class
        :param mainTerms: main terms of the cost function, for example similarity measure or collision detection
        :type mainTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :param regTerms: regularization terms of the cost function
        :type regTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :return: Tuple of displaced points and loss
        :rtype: (torch.Tensor,torch.Tensor)
        """
        if self.stationary:
            return self.callStationary(pts,velocityField,interpolator,mainTerms,regTerms)
        else:
            return self.callNonstationary(pts,velocityField,interpolator,mainTerms,regTerms)

    def callNonstationary(self,pts: Tensor,velocityField: Tensor,interpolator: Union[nearest,linear,cubic],mainTerms: list,regTerms: list)->Tuple[Tensor,Tensor]:
        """ Nonstationary numerical integration. See __call__.
        """
        if mainTerms is not None or regTerms is not None:
            loss=0
            pts_orig=pts.clone()
            for i in range(self.t_steps):
                vel,vel_jac,vel_lap=interpolator(pts,velocityField[i])
                dis=vel.t()*(1/self.t_steps)
                dis_jac=vel_jac*(1/self.t_steps)
                for term in regTerms:
                    loss=loss+term(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac)*(1/self.t_steps)
                pts = pts + dis
                #loss=loss+mainTerm(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac) #if data fidelty term is inside time integral in continuous formulation of the cost function
            for main in mainTerms:
                loss=loss+main(dis=pts-pts_orig)
            return pts,loss
        else:
            for i in range(self.t_steps):
                vel,jac,lap=interpolator(pts,velocityField[i])
                pts = pts + vel.t() * (1 / self.t_steps)
            return pts,None
    def callStationary(self,pts: Tensor,velocityField: Tensor,interpolator: Union[nearest,linear,cubic],mainTerms: list,regTerms: list)->Tuple[Tensor,Tensor]:
        """ Stationary numerical integration. See __call__.
        """
        if mainTerms is not None or regTerms is not None:
            loss=0
            pts_orig=pts.clone()
            for i in range(self.t_steps):
                vel,vel_jac,vel_lap=interpolator(pts,velocityField[0])
                dis=vel.t()*(1/self.t_steps)
                dis_jac=vel_jac*(1/self.t_steps)
                for term in regTerms:
                    loss=loss+term(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac)*(1/self.t_steps)
                pts = pts + dis
            for main in mainTerms:
                loss=loss+main(dis=pts-pts_orig)
            return pts,loss
        else:
            for i in range(self.t_steps):
                vel,jac,lap=interpolator(pts,velocityField[0])
                pts = pts + vel.t() * (1 / self.t_steps)
            return pts,None

class trapezoidal:
    """ Class for forward euler trapezoidal integration of velocity fields. Time interval is assumed to be [0;1] and step size is assumed to be equidistant.
    :param t_steps: steps taken to cover time interval
    :type t_steps: int
    :param stationary: flag for stationary velocity fields
    :type stationary: bool
    :param corrector_steps: maximum iterations for the predictor-corrector method
    :type corrector_steps: int
    :param tol: tolerance for predictor-corrector method
    :type tol: float
    """
    def __init__(self,t_steps: int,stationary=False,corrector_steps: int=1,tol: float=0.0001):
        """ Constructor method.
        """
        self.t_steps=t_steps
        self.stationary=stationary
        self.corrector_steps=corrector_steps
        self.tol=tol

    def __call__(self,pts: Tensor,velocityField: Tensor,interpolator: Union[nearest,linear,cubic],mainTerms: list,regTerms: list)->Tuple[Tensor,Tensor]:
        """ Numerical integration. Decides whether to treat the velocity field as stationary or non-stationary.
        :param pts: points coordinates in material configuration
        :type pts: torch.Tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.Tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: nearest, linear, or cubic interpolation class
        :param mainTerms: main terms of the cost function, for example similarity measure or collision detection
        :type mainTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :param regTerms: regularization terms of the cost function
        :type regTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :return: Tuple of displaced points and loss
        :rtype: (torch.Tensor,torch.Tensor)
        """
        if self.stationary:
            return self.callStationary(pts,velocityField,interpolator,mainTerms,regTerms)
        else:
            return self.callNonstationary(pts,velocityField,interpolator,mainTerms,regTerms)

    def callNonstationary(self,pts: Tensor,velocityField: Tensor,interpolator: Union[nearest,linear,cubic],mainTerms: list,regTerms: list)->Tuple[Tensor,Tensor]:
        """ Nonstationary numerical integration. See __call__.
        """
        if mainTerms is not None or regTerms is not None:
            loss=0
            pts_orig=pts.clone()
            for i in range(self.t_steps - 1):
                vel, vel_jac, vel_lap = interpolator(pts, velocityField[i])
                dis = vel.t() * (1 / (self.t_steps - 1))
                dis_jac=vel_jac*(1 / (self.t_steps - 1))
                pts_for = pts + dis
                pts_for_old = pts_for

                for j in range(self.corrector_steps):
                    vel_i_1, vel_jac_i_1, vel_lap_i_1 = interpolator(pts_for, velocityField[i + 1])
                    dis_i_1 = (vel.t() + vel_i_1.t()) * (1 / (self.t_steps - 1)) * 0.5
                    dis_jac_i_1 = (vel_jac+ vel_jac_i_1) * (1 / (self.t_steps - 1)) * 0.5
                    pts_for = pts + dis_i_1
                    for term in regTerms:
                        loss = loss + (term(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac)+term(vel=vel_i_1.t(), vel_jac=vel_jac_i_1, vel_lap=vel_lap_i_1, dis=dis_i_1, dis_jac=dis_jac_i_1))*0.5 * (1 / self.t_steps)
                    if torch.norm(pts_for.data - pts_for_old.data) < self.tol:
                        break
                    else:
                        pts_for_old = pts_for
                pts = pts_for
            for main in mainTerms:
                loss=loss+main(dis=pts-pts_orig)
            return pts,loss
        else:
            for i in range(self.t_steps-1):
                vel,vel_jac,vel_lap=interpolator(pts,velocityField[i])
                dis=vel.t()*(1 / (self.t_steps - 1))
                pts_for = pts + dis
                pts_for_old=pts_for
                for j in range(self.corrector_steps):
                    vel_i_1, vel_jac_i_1, vel_lap_i_1 = interpolator(pts_for, velocityField[i + 1])
                    dis_i_1 = (vel.t() + vel_i_1.t()) * (1 / (self.t_steps - 1)) * 0.5
                    pts_for = pts + dis_i_1
                    if torch.norm(pts_for.data - pts_for_old.data) < self.tol:
                        break
                    else:
                        pts_for_old = pts_for
                pts = pts_for
            return pts,None
    def callStationary(self,pts: Tensor,velocityField: Tensor,interpolator: Union[nearest,linear,cubic],mainTerms: list,regTerms: list)->Tuple[Tensor,Tensor]:
        """ Stationary numerical integration. See __call__.
        """
        if mainTerms is not None or regTerms is not None:
            loss = 0
            pts_orig=pts.clone()
            for i in range(self.t_steps - 1):
                vel, vel_jac, vel_lap = interpolator(pts, velocityField[0])
                dis = vel.t() * (1 / (self.t_steps - 1))
                dis_jac = vel_jac * (1 / (self.t_steps - 1))
                pts_for = pts + dis
                pts_for_old = pts_for
                for j in range(self.corrector_steps):
                    vel_i_1, vel_jac_i_1, vel_lap_i_1 = interpolator(pts_for, velocityField[0])
                    dis_i_1 = (vel.t() + vel_i_1.t()) * (1 / (self.t_steps - 1)) * 0.5
                    dis_jac_i_1 = (vel_jac + vel_jac_i_1) * (1 / (self.t_steps - 1)) * 0.5
                    pts_for = pts + dis_i_1
                    for term in regTerms:
                        loss = loss + (term(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac) + term(vel=vel_i_1.t(), vel_jac=vel_jac_i_1, vel_lap=vel_lap_i_1, dis=dis_i_1, dis_jac=dis_jac_i_1)) * 0.5 * (
                                           1 / self.t_steps)
                    if torch.norm(pts_for.data - pts_for_old.data) < self.tol:
                        break
                    else:
                        pts_for_old = pts_for
                pts = pts_for
            for main in mainTerms:
                loss=loss+main(dis=pts-pts_orig)
            return pts, loss
        else:
            for i in range(self.t_steps - 1):
                vel, vel_jac, vel_lap = interpolator(pts, velocityField[0])
                dis = vel.t() * (1 / self.t_steps)
                pts_for = pts + dis
                pts_for_old = pts_for
                for j in range(self.corrector_steps):
                    vel_i_1, vel_jac_i_1, vel_lap_i_1 = interpolator(pts_for, velocityField[0])
                    dis_i_1 = (vel.t() + vel_i_1.t()) * (1 / (self.t_steps - 1)) * 0.5
                    pts_for = pts + dis_i_1
                    if torch.norm(pts_for.data - pts_for_old.data) < self.tol:
                        break
                    else:
                        pts_for_old = pts_for
                pts = pts_for
            return pts, None


