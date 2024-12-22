import torch
import tqdm
from typing import Optional, Type, Union, Tuple
from torch import Tensor
from ..geometricTransformations import *
from .optim_gradientDescentBacktracking import *

class closureGradientDescent:
    """ Class for a custom closure function (see pytorch docs) to evaluate the loss function.
        :param optimizer: optimizer class used for the minimization problem
        :type optimizer: gradientDescentBacktracking or torch.optim.Optimizer class'
        :param transformer: transformer
        :type transformer: affineTransformation or nonrigidDeformation class
        :param mainTerms: main terms of the cost function, for example similarity measure or collision detection
        :type mainTerms: list of any regularization, collisionDetection, or similarityMeasure class
        :param regTerms: regularization terms of the cost function
        :type regTerms: list of any regularization, collisionDetection, or similarityMeasure class
            """
    def __init__(self,optimizer: Union[gradientDescentBacktracking,torch.optim.Optimizer],transformer: Union[affineTransformation,nonrigidDeformation],mainTerms: Optional[list]=None,regTerms: Optional[list]=[]):
        """Constructor method.                """
        self.optimizer=optimizer
        self.mainTerms=mainTerms
        self.regTerms=regTerms
        self.transformer=transformer

    def __call__(self,decisionVariables: Tensor)->Tensor:
        """ Initiates the forward pass. In case of gradients switched on, also zeros the gradients first.
                :param decisionVariables: Tensor with decision variables
                :type decisionVariables: torch.Tensor
                :return loss: backpropagated accumulated loss
                :rtype loss: torch.Tensor
                """
        if decisionVariables.requires_grad:
            self.optimizer.zero_grad()
            _,loss=self.transformer(decisionVariables,self.mainTerms,self.regTerms)
            loss.backward()
        else:
            _, loss = self.transformer(decisionVariables, self.mainTerms, self.regTerms)
        return loss

def algorithmGradientDescent(iterations: int,optimizerF: Union[gradientDescentBacktracking,torch.optim.Optimizer],closureF: Type,decisionVariables: Tensor)->dict:
    """ Starts a default iterative, gradient descent optimization scheme.
            :param iterations: number of steps in scheme
            :type iterations: int
            :param optimizerF: optimizer class used for the minimization problem
            :type optimizerF: gradientDescentBacktracking or torch.optim.Optimizer class
            :param closureF: closure function to calculate loss and backpropagate.
            :type closureF: either a function or a class with a __call__ method
            :param decisionVariables: Tensor with decision variables
            :type decisionVariables: torch.Tensor
            :return dict: dictionary with gradient history and objective function history
            :rtype dict: dict
            """
    gradientFHistory=[]
    objectiveLossFHistory=[]
    for i in tqdm.tqdm(range(iterations),desc="Progress"):
        loss=optimizerF.step(lambda: closureF(decisionVariables))
        gradientFHistory.append(torch.norm(decisionVariables.grad).item())
        objectiveLossFHistory.append(loss.detach().cpu().item())
        dict = {"gradientHistory": gradientFHistory,
            "objectiveLossHistory": objectiveLossFHistory}
    return dict


def algorithmGradientDescentStochastic(device: str,iterations: int,optimizerF: Union[gradientDescentBacktracking,torch.optim.Optimizer],closureF: Type,decisionVariables: Tensor,evalPoints: Tensor,evalPointsIntensities: Tensor,percentage: float,stochasticTerms: list)->dict:
    """ Starts a default iterative, gradient descent optimization scheme. It uses stochastic selection of evaluation points.
            :param device: computation device, see torch docs
            :type device: str
            :param iterations: number of steps in scheme
            :type iterations: int
            :param optimizerF: optimizer class used for the minimization problem
            :type optimizerF: gradientDescentBacktracking or torch.optim.Optimizer class
            :param closureF: closure function to calculate loss and backpropagate.
            :type closureF: either a function or a class with a __call__ method
            :param decisionVariables: Tensor with decision variables
            :type decisionVariables: torch.Tensor
            :param evalPoints: range of evaluation points available for selection
            :type evalPoints: torch.Tensor
            :param evalPointsIntensities: range of intensities of evaluation points available for selection
            :type evalPointsIntensities: torch.Tensor
            :param percentage: percentage of points used in each iteration ranging from [0;1]
            :type percentage: float
            :param stochasticTerms: list of mainTerms and regTerms that need updates to their points in every iteration
            :type stochasticTerms: list
            :return dict: dictionary with gradient history and objective function history
            :rtype dict: dict
            """
    gradientFHistory=[]
    objectiveLossFHistory=[]
    length=evalPoints.size(0)
    numberOfPoints=int(length*percentage)
    for i in tqdm.tqdm(range(iterations),desc="Progress"):
        random_tensor = torch.randperm(evalPoints.size(0))[:numberOfPoints]

        evalPointsStochastic = evalPoints[random_tensor].to(device=device)
        evalPointsIntensitiesStochastic = evalPointsIntensities[random_tensor].to(device=device)

        for term in stochasticTerms:
            if hasattr(term, 'pts'):
                term.pts=evalPointsStochastic
            if hasattr(term, 'intensities'):
                term.intensities=evalPointsIntensitiesStochastic

        loss=optimizerF.step(lambda: closureF(decisionVariables))
        gradientFHistory.append(torch.norm(decisionVariables.grad).item())
        objectiveLossFHistory.append(loss.detach().cpu().item())
        dict = {"gradientHistory": gradientFHistory,
            "objectiveLossHistory": objectiveLossFHistory}
    return dict
