import torch
import tqdm
from typing import Optional, Type, Union, Tuple
from torch import Tensor
from ..geometricTransformations import *
from .optim_gradientDescentBacktracking import *

class constrainerADMM:
    """Template class for ADMM coupling constraints.
                :param c: Tensor with additive constant
                :type c: torch.Tensor
                :param dualVariable: Tensor with dual variable
                :type dualVariable: torch.Tensor
                :param decisionVariablesXCoef: coefficient for X decision variable
                :type decisionVariablesXCoef: float
                :param decisionVariablesZCoef: coefficient for X decision variable
                :type decisionVariablesZCoef: float"""
    def __init__(self,c: Tensor,dualVariable: Tensor,decisionVariablesXCoef:float,decisionVariablesZCoef:float):
        """Constructor method."""
        self.c=c
        self.dualVariable=dualVariable
        self.decisionVariablesXCoef=decisionVariablesXCoef
        self.decisionVariablesZCoef=decisionVariablesZCoef
    def __call__(self,decisionVariable: Tensor,decisionVariableFixed: Tensor,decisionVariableCoef: float,decisionVariableFixedCoef: float):
        pass
    def updateDualVariable(self, decisionVariablesX: Tensor, decisionVariablesZ: Tensor):
        pass

class constrainerEulerianADMM(constrainerADMM):
    """Class for Eulerian ADMM coupling constraints. See Template class.
    """
    def __call__(self, decisionVariables: Tensor,decisionVariablesCoef: float,fixedDecisionVariables: Tensor,fixedDecisionVariablesCoef: float)->Tensor:
        """Calculates constrainer loss.
                :param decisionVariables: Tensor with decision variables
                :type decisionVariables: torch.Tensor
                :param decisionVariablesCoef: coefficient for decision variables
                :type decisionVariablesCoef: float
                :param fixedDecisionVariables: Tensor with fixed decision variables
                :type fixedDecisionVariables: torch.Tensor
                :param fixedDecisionVariablesCoef: coefficient for fixed decision variables
                :type fixedDecisionVariablesCoef: float
                :return : constrainer loss
                :rtype : torch.Tensor
        """
        return decisionVariablesCoef * decisionVariables + fixedDecisionVariablesCoef * fixedDecisionVariables - self.c +self.dualVariable
    def updateDualVariable(self, decisionVariablesX: Tensor, decisionVariablesZ: Tensor):
        """Updates the dual variable.
                :param decisionVariablesX: Tensor with decision variables
                :type decisionVariablesX: torch.Tensor
                :param decisionVariablesZ: Tensor with decision variables
                :type decisionVariablesZ: torch.Tensor
        """
        self.dualVariable = self.dualVariable + self.decisionVariablesXCoef * decisionVariablesX.data + self.decisionVariablesZCoef * decisionVariablesZ.data - self.c
class constrainerLagrangianADMM(constrainerADMM):
    """Class for Lagrangian ADMM coupling constraints. See Template class.
        :param transformer: transformer
        :type transformer: affineTransformation or nonrigidDeformation class
        :param pts: coupling points
        :type pts: torch.Tensor
    """
    def __init__(self,c: Tensor,dualVariable: Tensor,decisionVariablesXCoef: float,decisionVariablesZCoef: float,transformer: Union[affineTransformation,nonrigidDeformation],pts: Tensor):
        super().__init__(c,dualVariable,decisionVariablesXCoef,decisionVariablesZCoef)
        self.transformer=transformer
        self.pts=pts
    def __call__(self, decisionVariables: Tensor,decisionVariablesCoef: float,fixedDecisionVariables: Tensor,fixedDecisionVariablesCoef: float)->Tensor:
        """Calculates constrainer loss.
                :param decisionVariables: Tensor with decision variables
                :type decisionVariables: torch.Tensor
                :param decisionVariablesCoef: coefficient for decision variables
                :type decisionVariablesCoef: float
                :param fixedDecisionVariables: Tensor with fixed decision variables
                :type fixedDecisionVariables: torch.Tensor
                :param fixedDecisionVariablesCoef: coefficient for fixed decision variables
                :type fixedDecisionVariablesCoef: float
                :return : constrainer loss
                :rtype : torch.Tensor
        """
        pts=self.transformer.apply(self.pts,decisionVariables)
        ptsFixed=self.transformer.apply(self.pts,fixedDecisionVariables)
        return decisionVariablesCoef * pts + fixedDecisionVariablesCoef * ptsFixed - self.c +self.dualVariable
    def updateDualVariable(self, decisionVariablesX: Tensor, decisionVariablesZ: Tensor):
        """Updates the dual variable.
                :param decisionVariablesX: Tensor with decision variables
                :type decisionVariablesX: torch.Tensor
                :param decisionVariablesZ: Tensor with decision variables
                :type decisionVariablesZ: torch.Tensor
        """
        ptsX=self.transformer.apply(self.pts,decisionVariablesX.data)
        ptsZ=self.transformer.apply(self.pts,decisionVariablesZ.data)
        self.dualVariable = self.dualVariable + self.decisionVariablesXCoef * ptsX + self.decisionVariablesZCoef * ptsZ - self.c

class closureADMM():
    """ Class for a custom closure function (see pytorch docs) to evaluate the loss function, specifically for ADMM algorithm.
            :param optimizer: optimizer class used for the minimization problem
            :type optimizer: torch.optim.optimizer class or custom class
            :param decisionVariablesCoef: coefficient for decisionVariables
            :type decisionVariablesCoef: int, float ot torch.Tensor
            :param decisionVariablesFixedCoef: coefficient for fixed decisionVariables
            :type decisionVariablesFixedCoef: int, float ot torch.Tensor
            :param rho: penalty parameter to weigh the trade off,see ADMM
            :type rho: int or float
            :param mainTerms: main term to be minimized
            :type mainTerms: simMeasure or regularizer class
            :param mainTermCoef: coefficient for mainTerm
            :type mainTermCoef: either int or float
            :param regTerms: list of regularizers to constrain the minimization problem
            :type regTerms: list of regularizer classes
            :param regTermsCoefs: list of coefficients for the regularizers
            :type regTermsCoefs: list of ints or floats
            :return: backpropagated accumulated loss
            :rtype: torch.Tensor
            """
    def __init__(self,optimizer: Union[gradientDescentBacktracking,torch.optim.Optimizer],transformer: Union[affineTransformation,nonrigidDeformation],rho: Optional[float]=1.0,mainTerms: Optional[list]=None, regTerms: Optional[list]=[]):
        """Constructor method.            """
        self.optimizer=optimizer
        self.rho=rho
        self.mainTerms=mainTerms
        self.regTerms=regTerms
        self.transformer=transformer

    def __call__(self,decisionVariables: Tensor,decisionVariablesCoef: float,fixedDecisionVariables: Tensor,fixedDecisionVariablesCoef: float,constrainer: Union[Type[constrainerEulerianADMM],Type[constrainerLagrangianADMM]])->Tensor:
        """ Initiates the forward pass for ADMM. In case of gradients switched on, also zeros the gradients first.
                :param decisionVariables: tensor with decision variables
                :type decisionVariables: torch.Tensor
                :param decisionVariablesCoef: coefficients for decision variables
                :type decisionVariablesCoef: float
                :param fixedDecisionVariables: tensor with fixed decision variables
                :type fixedDecisionVariables: torch.Tensor
                :param fixedDecisionVariablesCoef: coefficients for decision variables
                :type fixedDecisionVariablesCoef: float
                :param constrainer: measures the deviation from constraints, see ADMM
                :type constrainer: constrainerEulerianADMM or constrainerLagrangianADMM class
                :return: backpropagated accumulated loss
                :rtype: torch.Tensor
                """
        if decisionVariables.requires_grad:
            self.optimizer.zero_grad()
            _,loss=self.transformer(decisionVariables,self.mainTerms,self.regTerms)
            loss=loss+0.5*self.rho*(torch.norm(constrainer(decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef))**2)
            loss.backward()
        else:
            _, loss = self.transformer(decisionVariables, self.mainTerms, self.regTerms)
            loss=loss+0.5*self.rho*(torch.norm(constrainer(decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef))**2)
        return loss
def algorithmADMM(iterations: int,subsolverIterations: int,constrainer: Union[constrainerEulerianADMM,constrainerLagrangianADMM],optimizerF: Union[gradientDescentBacktracking,torch.optim.Optimizer],optimizerG: Union[gradientDescentBacktracking,torch.optim.Optimizer],closureF: Type,closureG: Type,decisionVariablesX: Tensor,decisionVariablesZ: Tensor,rho: float)->dict:
    """ Starts the ADMM iterative optimization scheme.
            :param iterations: number of steps in scheme
            :type iterations: int
            :param subsolverIterations: number of steps for every subsolver
            :type subsolverIterations: int
            :param constrainer: measures the deviation from constraints, see ADMM
            :type constrainer: constrainerEulerianADMM or constrainerLagrangianADMM class
            :param optimizerF: optimizer class used for the first minimization problem
            :type optimizerF: gradientDescentBacktracking or torch.optim.Optimizer class
            :param optimizerG: optimizer class used for the second minimization problem
            :type optimizerG: gradientDescentBacktracking or torch.optim.Optimizer class
            :param closureF: closure function to calculate loss and backpropagate for the first minimization problem.
            :type closureF: either a function or a class with a __call__ method
            :param closureG: closure function to calculate loss and backpropagate for the second minimization problem.
            :type closureG: either a function or a class with a __call__ method
            :param decisionVariablesX: Tensor with decision variables for the first problem
            :type decisionVariablesX: torch.Tensor
            :param decisionVariablesZ: Tensor with decision variables for the second problem
            :type decisionVariablesZ: torch.Tensor'
            :param rho: penalty parameter
            :type rho: float
            :return dict: dictionary with gradient history and objective function history
            :rtype dict: dict
            """
    objectiveLossFHistory=[]
    objectiveLossGHistory=[]
    primalResidualHistory=[]
    dualResidualHistory=[]
    dualVariableHistory=[]
    for i in tqdm.tqdm(range(iterations),desc="Progress"):
        z_old=decisionVariablesZ.data.clone()
        for j in range(subsolverIterations):
            lossX=optimizerF.step(lambda: closureF(decisionVariablesX,constrainer.decisionVariablesXCoef,decisionVariablesZ.data,constrainer.decisionVariablesZCoef,constrainer))
        objectiveLossFHistory.append(lossX.cpu().item())
        for j in range(subsolverIterations):
            lossZ=optimizerG.step(lambda: closureG(decisionVariablesZ,constrainer.decisionVariablesZCoef,decisionVariablesX.data,constrainer.decisionVariablesXCoef,constrainer))
        constrainer.updateDualVariable(decisionVariablesX,decisionVariablesZ)
        objectiveLossGHistory.append(lossZ.cpu().item())
        primalResidualHistory.append(torch.norm((decisionVariablesX.data - decisionVariablesZ.data)).cpu().item())
        dualResidualHistory.append(torch.norm(-rho * (decisionVariablesZ.data - z_old)).cpu().item())
        dualVariableHistory.append(torch.norm(constrainer.dualVariable).cpu().item())
    dict = {"objectiveLossFHistory": objectiveLossFHistory,
            "objectiveLossGHistory": objectiveLossGHistory,
            "primalResidualHistory": primalResidualHistory,
            "dualResidualHistory": dualResidualHistory,
            "dualVariableHistory": dualVariableHistory
            }
    return dict



def algorithmADMMStochastic(device: str,iterations: int,subsolverIterations: int,constrainer: Union[constrainerEulerianADMM,constrainerLagrangianADMM],optimizerF: Union[gradientDescentBacktracking,torch.optim.Optimizer],optimizerG: Union[gradientDescentBacktracking,torch.optim.Optimizer],closureF: Type,closureG: Type,decisionVariablesX: Tensor,decisionVariablesZ: Tensor,rho: float,evalPoints1: Tensor,evalPointsIntensities1: Tensor,percentage1: float,stochasticTerms1: list,evalPoints2: Tensor,evalPointsIntensities2: Tensor,percentage2: float,stochasticTerms2: list,pointsMaskLabel1: Optional[Tensor]=None,pointsMaskLabel2: Optional[Tensor]=None)->dict:
    """ Starts the ADMM iterative optimization scheme.  It uses stochastic selection of evaluation points.
            :param device: computation device, see torch docs
            :type device: str
            :param iterations: number of steps in scheme
            :type iterations: int
            :param subsolverIterations: number of steps for every subsolver
            :type subsolverIterations: int
            :param constrainer: measures the deviation from constraints, see ADMM
            :type constrainer: constrainerEulerianADMM or constrainerLagrangianADMM class
            :param optimizerF: optimizer class used for the first minimization problem
            :type optimizerF: gradientDescentBacktracking or torch.optim.Optimizer class
            :param optimizerG: optimizer class used for the second minimization problem
            :type optimizerG: gradientDescentBacktracking or torch.optim.Optimizer class
            :param closureF: closure function to calculate loss and backpropagate for the first minimization problem.
            :type closureF: either a function or a class with a __call__ method
            :param closureG: closure function to calculate loss and backpropagate for the second minimization problem.
            :type closureG: either a function or a class with a __call__ method
            :param decisionVariablesX: Tensor with decision variables for the first problem
            :type decisionVariablesX: torch.Tensor
            :param decisionVariablesZ: Tensor with decision variables for the second problem
            :type decisionVariablesZ: torch.Tensor'
            :param rho: penalty parameter
            :type rho: float
            :param evalPoints1: range of evaluation points available for selection for the first problem
            :type evalPoints1: torch.Tensor
            :param evalPointsIntensities1: range of intensities of evaluation points available for selection for the first problem
            :type evalPointsIntensities1: torch.Tensor
            :param percentage1: percentage of points used in each iteration ranging from [0;1] for the first problem
            :type percentage1: float
            :param stochasticTerms1: list of mainTerms and regTerms that need updates to their points in every iteration for the first problem
            :type stochasticTerms1: list
            :param pointsMaskLabel1: points mask labels for points that need to be exchanged for the first problem
            :type pointsMaskLabel1: torch.Tensor
            :param evalPoints2: range of evaluation points available for selection for the second problem
            :type evalPoints2: torch.Tensor
            :param evalPointsIntensities2: range of intensities of evaluation points available for selection for the second problem
            :type evalPointsIntensities2: torch.Tensor
            :param percentage2: percentage of points used in each iteration ranging from [0;1] for the second problem
            :type percentage2: float
            :param stochasticTerms2: list of mainTerms and regTerms that need updates to their points in every iteration for the second problem
            :type stochasticTerms2: list
            :param pointsMaskLabel2: points mask labels for points that need to be exchanged for the second problem
            :type pointsMaskLabel2: torch.Tensor
            :return dict: dictionary with gradient history and objective function history
            :rtype dict: dict
            """
    objectiveLossFHistory=[]
    objectiveLossGHistory=[]
    primalResidualHistory=[]
    dualResidualHistory=[]
    dualVariableHistory=[]
    if stochasticTerms1 is not None:
        length1=evalPoints1.size(0)
        numberOfPoints1=int(length1*percentage1)
    if stochasticTerms2 is not None:
        length2=evalPoints2.size(0)
        numberOfPoints2=int(length2*percentage2)
    for i in tqdm.tqdm(range(iterations),desc="Progress"):
        if stochasticTerms1 is not None:
            random_tensor1 = torch.randperm(evalPoints1.size(0))[:numberOfPoints1]
            evalPointsStochastic1 = evalPoints1[random_tensor1].to(device=device)
            evalPointsIntensitiesStochastic1 = evalPointsIntensities1[random_tensor1].to(device=device)
            for cnt,term in enumerate(stochasticTerms1):
                if hasattr(term, 'pts'):
                    if hasattr(term, 'pointsMask'):
                        if term.pointsMask is not None:
                            term.pts[term.pointsMask==pointsMaskLabel1[cnt]]=evalPointsStochastic1
                        else:
                            term.pts=evalPointsStochastic1
                    else:
                        term.pts=evalPointsStochastic1
                if hasattr(term, 'intensities'):
                    term.intensities=evalPointsIntensitiesStochastic1
        if stochasticTerms2 is not None:
            random_tensor2 = torch.randperm(evalPoints2.size(0))[:numberOfPoints2]
            evalPointsStochastic2 = evalPoints2[random_tensor2].to(device=device)
            evalPointsIntensitiesStochastic2 = evalPointsIntensities2[random_tensor2].to(device=device)
            for term in stochasticTerms2:
                if hasattr(term, 'pts'):
                    if hasattr(term, 'pts'):
                        if hasattr(term, 'pointsMask'):
                            if term.pointsMask is not None:
                                term.pts[term.pointsMask == pointsMaskLabel2[cnt]] = evalPointsStochastic2
                            else:
                                term.pts = evalPointsStochastic2
                        else:
                            term.pts = evalPointsStochastic2
                    if hasattr(term, 'intensities'):
                        term.intensities = evalPointsIntensitiesStochastic2
        z_old=decisionVariablesZ.data.clone()
        for j in range(subsolverIterations):
            lossX=optimizerF.step(lambda: closureF(decisionVariablesX,constrainer.decisionVariablesXCoef,decisionVariablesZ.data,constrainer.decisionVariablesZCoef,constrainer))

        objectiveLossFHistory.append(lossX.cpu().item())
        for j in range(subsolverIterations):
            lossZ=optimizerG.step(lambda: closureG(decisionVariablesZ,constrainer.decisionVariablesZCoef,decisionVariablesX.data,constrainer.decisionVariablesXCoef,constrainer))

        constrainer.updateDualVariable(decisionVariablesX,decisionVariablesZ)
        objectiveLossGHistory.append(lossZ.cpu().item())
        primalResidualHistory.append(torch.norm((decisionVariablesX.data - decisionVariablesZ.data)).cpu().item())
        dualResidualHistory.append(torch.norm(-rho * (decisionVariablesZ.data - z_old)).cpu().item())
        dualVariableHistory.append(torch.norm(constrainer.dualVariable).cpu().item())
    dict = {"objectiveLossFHistory": objectiveLossFHistory,
            "objectiveLossGHistory": objectiveLossGHistory,
            "primalResidualHistory": primalResidualHistory,
            "dualResidualHistory": dualResidualHistory,
            "dualVariableHistory": dualVariableHistory
            }
    return dict
