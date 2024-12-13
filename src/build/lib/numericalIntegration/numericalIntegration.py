import torch
class forwardEulerNumInt():
    """ Class for forward euler numerical integration of velocity fields; time interval is assumed [0;1]; step size is assumed to be equidistant
    :param t_steps: steps taken to cover time interval
    :type t_steps: int
    :param stationary: flag for stationary velocity fields
    :type stationary: boolean
    """
    def __init__(self,t_steps,stationary=True):
        """ constructor method
        """
        self.t_steps=t_steps
        self.stationary=stationary

    def __call__(self,pts,velocityField,interpolator,mainTerm,regTerms):
        """ Numerical integration.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: interpolation class
        :param mainTerm: main term of cost function to be evaluated
        :type mainTerm: custom class
        :param regTerms: regularization terms of cost function to be evaluated on the fly
        :type regTerms: custom class
        """
        if self.stationary:
            return self.callStationary(pts,velocityField,interpolator,mainTerm,regTerms)
        else:
            return self.callNonstationary(pts,velocityField,interpolator,mainTerm,regTerms)

    def callNonstationary(self,pts,velocityField,interpolator,mainTerm,regTerms):
        """ Nonstationary numerical integration.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: interpolation class
        :param mainTerm: main term of cost function to be evaluated
        :type mainTerm: custom class
        :param regTerms: regularization terms of cost function to be evaluated on the fly
        :type regTerms: custom class
        :return: displaced points as tensor (#points,dim) and accumulated loss function
        :rtype: torch.tensor  & torch.tensor
        """
        if mainTerm is not None or regTerms is not None:
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
            for main in mainTerm:
                loss=loss+main(dis=pts-pts_orig)
            return pts,loss
        else:
            for i in range(self.t_steps):
                vel,jac,lap=interpolator(pts,velocityField[i])
                pts = pts + vel.t() * (1 / self.t_steps)
            return pts,None
    def callStationary(self,pts,velocityField,interpolator,mainTerm,regTerms):
        """ Stationary numerical integration.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: interpolation class
        :param mainTerm: main term of cost function to be evaluated
        :type mainTerm: custom class
        :param regTerms: regularization terms of cost function to be evaluated on the fly
        :type regTerms: custom class
        :return: displaced points as tensor (#points,dim) and accumulated loss function
        :rtype: torch.tensor & torch.tensor
        """
        if mainTerm is not None or regTerms is not None:
            loss=0
            pts_orig=pts.clone()
            for i in range(self.t_steps):
                vel,vel_jac,vel_lap=interpolator(pts,velocityField[0])
                dis=vel.t()*(1/self.t_steps)
                dis_jac=vel_jac*(1/self.t_steps)
                for term in regTerms:
                    loss=loss+term(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac)*(1/self.t_steps)
                pts = pts + dis
            for main in mainTerm:
                loss=loss+main(dis=pts-pts_orig)
            return pts,loss
        else:
            for i in range(self.t_steps):
                vel,jac,lap=interpolator(pts,velocityField[0])
                pts = pts + vel.t() * (1 / self.t_steps)
            return pts,None



class trapezoidalNumInt():
    """ Class for trapezoidal  euler numerical integration of velocity fields.
    Time interval is assumed [0;1].
    Step size is assumed to be equidistant.
    Uses forward euler as first predictor in predictor–corrector method.
    :param t_steps: steps taken to cover time interval
    :type t_steps: int
    :param stationary: flag for stationary velocity fields
    :type stationary: boolean
    :param corrector_steps: iterations for the predictor-corrector method
    :type corrector_steps: int
    :param tol: tolerance for predictor-corrector method
    :type tol: float
    """
    def __init__(self,t_steps,stationary=True,corrector_steps=1,tol=0.0001):
        """ constructor method
        """
        self.t_steps=t_steps
        self.stationary=stationary
        self.corrector_steps=corrector_steps
        self.tol=tol

    def __call__(self,pts,velocityField,interpolator,mainTerm,regTerms):
        """ Numerical integration.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :param spline: spline interpolation object
        :param evalFunc: regularization object
        :return: either displaced points as tensor (#points,dim) or evalFunc value
        :param integrateProperty: declares if the eval property needs to be integrated or not
        :type integrateProperty: boolean
        """
        if self.stationary:
            return self.callStationary(pts,velocityField,interpolator,mainTerm,regTerms)
        else:
            return self.callNonstationary(pts,velocityField,interpolator,mainTerm,regTerms)

    def callNonstationary(self,pts,velocityField,interpolator,mainTerm,regTerms):
        """ Nonstationary numerical integration.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: interpolation class
        :param mainTerm: main term of cost function to be evaluated
        :type mainTerm: custom class
        :param regTerms: regularization terms of cost function to be evaluated on the fly
        :type regTerms: custom class
        :return: displaced points as tensor (#points,dim) and accumulated loss function
        :rtype: torch.tensor  torch.tensor
        """
        if mainTerm is not None or regTerms is not None:
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
                    #this term if sim measure is evaluted only at end, meaning not time integrted
                    #loss=loss+(mainTerm(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac)+mainTerm(vel=vel_i_1.t(), vel_jac=vel_jac_i_1, vel_lap=vel_lap_i_1, dis=dis_i_1, dis_jac=dis_jac_i_1))*0.5 * (1 / self.t_steps)#if data fidelty term is inside time integral in continuous formulation of the cost function
                    if torch.norm(pts_for.data - pts_for_old.data) < self.tol:
                        break
                    else:
                        pts_for_old = pts_for
                pts = pts_for
            for main in mainTerm:
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
    def callStationary(self,pts,velocityField,interpolator,mainTerm,regTerms):
        """ Stationary numerical integration.
        :param pts: points as tensor (#points,dim) of cartesian coordinates to be displaced
        :type pts: torch.tensor
        :param velocityField: velocity field (time frames, dimensions, # control points dim 1, # control points dim 2 [,# control points dim 3])
        :type velocityField: torch.tensor
        :param interpolator:  interpolation method for velocity fields
        :type interpolator: interpolation class
        :param mainTerm: main term of cost function to be evaluated
        :type mainTerm: custom class
        :param regTerms: regularization terms of cost function to be evaluated on the fly
        :type regTerms: custom class
        :return: displaced points as tensor (#points,dim) and accumulated loss function
        :rtype: torch.tensor  torch.tensor
        """
        if mainTerm is not None or regTerms is not None:
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
                    # this term if sim measure is evaluted only at end, meaning not time integrted
                    #loss=loss+(mainTerm(vel=vel.t(),vel_jac=vel_jac,vel_lap=vel_lap,dis=dis,dis_jac=dis_jac)+mainTerm(vel=vel_i_1.t(), vel_jac=vel_jac_i_1, vel_lap=vel_lap_i_1, dis=dis_i_1, dis_jac=dis_jac_i_1))*0.5 * (1 / self.t_steps)#if data fidelty term is inside time integral in continuous formulation of the cost function
                    if torch.norm(pts_for.data - pts_for_old.data) < self.tol:
                        break
                    else:
                        pts_for_old = pts_for
                pts = pts_for
            for main in mainTerm:
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



class trapezoidalNumIntDetailed():
    def __init__(self,t_steps,stationary=True,corrector_steps=1,tol=0.0001,detailed=False,device=None):
        """ constructor method
        """
        self.t_steps=t_steps
        self.stationary=stationary
        self.corrector_steps=corrector_steps
        self.tol=tol
        self.detailed=detailed
        self.device=device

    def __call__(self,pts,velocityField,interpolator,mainTerm,regTerms):
        if self.detailed:
            pass
        else:
            return self.callNonstationary(pts,velocityField,interpolator,mainTerm,regTerms)

    def callNonstationary(self,pts,velocityField,interpolator,mainTerm,regTerms):

        trajectory=torch.zeros((pts.size(0)*(self.t_steps-1),3),device=self.device)
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
            if i==0:
                trajectory=pts_for.clone()
            else:
                trajectory=torch.cat((trajectory,pts_for.clone()))
            pts = pts_for
        return pts,trajectory
