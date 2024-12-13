import torch
import tqdm
class closureGradientDescent():
    """ Class for a custom closure function (see pytorch docs) to evaluate the loss function.
            :param optimizer: optimizer class used for the minimization problem
            :type optimizer: torch.optim.optimizer class
            :param mainTerm: main term that consitutes the objective function
            :type mainTerm: custom class
            :param mainTermCoef: coefficient for mainTerm
            :type mainTermCoef: either int or float
            :param regTerms: list of regularizers to constrain the minimization problem
            :type regTerms: list of regularizer classes MUST BE A LIST
            :param regTermsCoefs: list of coefficients for the regularizers
            :type regTermsCoefs: list of ints or floats
            """
    def __init__(self,optimizer,transformer,mainTerm=None,regTerms=[]):
        """Constructor method
                """
        self.optimizer=optimizer
        self.mainTerm=mainTerm
        self.regTerms=regTerms
        self.transformer=transformer

    def __call__(self,decisionVariables):
        """ Calculates the loss function by forward passing the similarity measure and a number of regularization terms and computing the gradients.
                :param decisionVariables: tensor with decision variables
                :type decisionVariables: torch.Tensor with gradient True
                :return: backpropagated accumulated loss
                :rtype: torch.Tensor
                """
        if decisionVariables.requires_grad:
            self.optimizer.zero_grad()
            _,loss=self.transformer(decisionVariables,self.mainTerm,self.regTerms)
            loss.backward()
        else:
            _, loss = self.transformer(decisionVariables, self.mainTerm, self.regTerms)
        return loss
def algorithmGradientDescent(iterations,optimizerF,closureF,decisionVariables):
    """ Starts a default iterative optimization scheme/algorithm. Decides whether step() is called in its default or backtracking implementation (only for optimizer_gradientDescentBT).
            :param iterations: number of steps in scheme
            :type iterations: int
            :param optimizerF: optimizer class used for the minimization problem
            :type optimizerF: torch.optim.optimizer class
            :param closureF: closure function to calculate loss and backpropagte
            :type closureF: closure class
            :param decisionVariables: tensor with independent variables or parameters to be optimized
            :type decisionVariables: torch.Tensor with gradient True
            :return: dictionary with gradient history and objective function history
            :rtype: dict
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


def algorithmGradientDescentStochastic(device,iterations,optimizerF,closureF,decisionVariables,evalPoints,evalPointsIntensities,percentage,stochasticTerms):
    """ Starts a default iterative optimization scheme/algorithm. Decides whether step() is called in its default or backtracking implementation (only for optimizer_gradientDescentBT).
            :param iterations: number of steps in scheme
            :type iterations: int
            :param optimizerF: optimizer class used for the minimization problem
            :type optimizerF: torch.optim.optimizer class
            :param closureF: closure function to calculate loss and backpropagte
            :type closureF: closure class
            :param decisionVariables: tensor with independent variables or parameters to be optimized
            :type decisionVariables: torch.Tensor with gradient True
            :return: dictionary with gradient history and objective function history
            :rtype: dict
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
