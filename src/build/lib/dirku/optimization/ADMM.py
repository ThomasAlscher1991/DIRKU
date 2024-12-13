import torch
import tqdm
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
            :param mainTerm: main term to be minimized
            :type mainTerm: simMeasure or regularizer class
            :param mainTermCoef: coefficient for mainTerm
            :type mainTermCoef: either int or float
            :param regTerms: list of regularizers to constrain the minimization problem
            :type regTerms: list of regularizer classes
            :param regTermsCoefs: list of coefficients for the regularizers
            :type regTermsCoefs: list of ints or floats
            :return: backpropagated accumulated loss
            :rtype: torch.Tensor
            """
    def __init__(self,optimizer,transformer,rho=None,mainTerm=None, regTerms=[]):
        """Constructor method
                """
        self.optimizer=optimizer
        self.rho=rho
        self.mainTerm=mainTerm
        self.regTerms=regTerms
        self.transformer=transformer

    def __call__(self,decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef,constrainer):
        """ Calculates the loss function by forward passing the similarity measure and a number of regularization terms and computing the gradients.
                :param decisionVariables: tensor with decision variables
                :type decisionVariables: torch.Tensor with gradient True
                :param fixedDecisionVariables: tensor with decision variables fixed in this iteration and treated as constants
                :type fixedDecisionVariables: torch.Tensor with gradient True
                :param dualVariable: measures the deviation from constraints, see ADMM
                :type dualVariable: torch.Tensor
                :return: backpropagated accumulated loss
                :rtype: torch.Tensor
                """
        if decisionVariables.requires_grad:
            self.optimizer.zero_grad()
            _,loss=self.transformer(decisionVariables,self.mainTerm,self.regTerms)
            loss=loss+0.5*self.rho*(torch.norm(constrainer(decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef))**2)
            loss.backward()
        else:
            _, loss = self.transformer(decisionVariables, self.mainTerm, self.regTerms)
            loss=loss+0.5*self.rho*(torch.norm(constrainer(decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef))**2)
        return loss
def algorithmADMM(iterations,subsolverIterations,constrainer,optimizerF,optimizerG,closureF,closureG,decisionVariablesX,decisionVariablesZ,rho):
    """ Starts the ADMM iterative optimization scheme/algorithm.
        Requires optimization problem to be split into two main terms and any number of regularization terms.
        Decides whether step() is called in its default or backtracking implementation (only for optimizer_gradientDescentBT) for both optimizers.
            :param iterations: number of steps in overall ADMM scheme
            :type iterations: int
            :param subsolverIterations: number of steps in each optimizer scheme
            :type subsolverIterations: int
            :param optimizerF: optimizer class used for the minimization problem for x
            :type optimizerF: torch.optim.optimizer class
            :param optimizerG: optimizer class used for the minimization problem for z
            :type optimizerG: torch.optim.optimizer class
            :param closureF: closure function to calculate loss and backpropagte for x
            :type closureF: closure class
            :param closureG: closure function to calculate loss and backpropagte for z
            :type closureG: closure class
            :param decisionVariablesX: tensor with independent variables or parameters to be optimized
            :type decisionVariablesX: torch.Tensor with gradient True
            :param decisionVariablesZ: tensor with independent variables or parameters to NOT be optimized
            :type decisionVariablesZ: torch.Tensor with gradient True
            :param rho: penalty parameter to weigh the trade off,see ADMM
            :type rho: int or float
            :return: dictionary with history of both objective functions, dual and primal residual, and dual variable
            :rtype: dict
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
        ###
        objectiveLossFHistory.append(lossX.cpu().item())


        for j in range(subsolverIterations):

            lossZ=optimizerG.step(lambda: closureG(decisionVariablesZ,constrainer.decisionVariablesZCoef,decisionVariablesX.data,constrainer.decisionVariablesXCoef,constrainer))
        ###



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



def algorithmADMMStochasticTwoSet(device,iterations,subsolverIterations,constrainer,optimizerF,optimizerG,closureF,closureG,decisionVariablesX,decisionVariablesZ,rho,evalPoints1,evalPointsIntensities1,percentage1,stochasticTerms1,evalPoints2,evalPointsIntensities2,percentage2,stochasticTerms2,pointsMaskLabel1=None,pointsMaskLabel2=None):
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

class constrainerADMM():
    def __init__(self,c,dualVariable,decisionVariablesXCoef,decisionVariablesZCoef):
        self.c=c
        self.dualVariable=dualVariable
        self.decisionVariablesXCoef=decisionVariablesXCoef
        self.decisionVariablesZCoef=decisionVariablesZCoef


    def __call__(self,decisionVariable,decisionVariableFixed,decisionVariableCoef,decisionVariableFixedCoef):
        pass

    def updateDualVariable(self, decisionVariablesX, decisionVariablesZ):
        pass


class constrainerEulerianADMM(constrainerADMM):
    def __call__(self, decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef):
        return decisionVariablesCoef * decisionVariables + fixedDecisionVariablesCoef * fixedDecisionVariables - self.c +self.dualVariable
    def updateDualVariable(self, decisionVariablesX, decisionVariablesZ):
        self.dualVariable = self.dualVariable + self.decisionVariablesXCoef * decisionVariablesX.data + self.decisionVariablesZCoef * decisionVariablesZ.data - self.c

class constrainerLagrangianADMM(constrainerADMM):
    def __init__(self,c,dualVariable,decisionVariablesXCoef,decisionVariablesZCoef,transformer,pts):
        super().__init__(c,dualVariable,decisionVariablesXCoef,decisionVariablesZCoef)
        self.transformer=transformer
        self.pts=pts


    def __call__(self, decisionVariables,decisionVariablesCoef,fixedDecisionVariables,fixedDecisionVariablesCoef):
        pts=self.transformer.apply(self.pts,decisionVariables)
        ptsFixed=self.transformer.apply(self.pts,fixedDecisionVariables)

        return decisionVariablesCoef * pts + fixedDecisionVariablesCoef * ptsFixed - self.c +self.dualVariable

    def updateDualVariable(self, decisionVariablesX, decisionVariablesZ):
        ptsX=self.transformer.apply(self.pts,decisionVariablesX.data)
        ptsZ=self.transformer.apply(self.pts,decisionVariablesZ.data)
        self.dualVariable = self.dualVariable + self.decisionVariablesXCoef * ptsX + self.decisionVariablesZCoef * ptsZ - self.c