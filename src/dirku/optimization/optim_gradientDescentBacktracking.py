from torch.optim.optimizer import Optimizer
import torch



class gradientDescentBacktracking(Optimizer):
    """ Class for a custom gradient descent optimizer with backtracking line search from Armijo (1966), heavily derived from pytorch's Optimizers.
        It requires for the __call__ method to be passed an argument nograd, which is the closure.nograd()method with the independent variable set to .data.
            :param params: FROM  PYTORCH - an iterable of torch.Tensor s or dict s. Specifies what Tensors should be optimized.
            :type params: torch.Tensor
            :param lr: learning rate
            :type lr: float
            :param max_iters: maximum iterations for the backtracking line search
            :type max_iters: int
            :param alpha: step size reduction coefficient (0;1)
            :type alpha: float
            """
    def __init__(self, params, lr=0.1,max_iters=1,c=0.0004,alpha=0.5):
        """ Constructor method.
                """
        defaults = dict(custom_parameter=lr)
        super(gradientDescentBacktracking, self).__init__(params, defaults)
        self.loss_old=None
        self.max_iters=max_iters
        self.c=c
        self.alpha=alpha

    def step(self, closure=None):
        """FROM PYTORCH - Performs a single optimization step.
            :param closure: closure function to calculate loss and backpropagte.
            :type closure: closure class
            :return: accumulated loss
            :rtype: torch.Tensor
            """
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            custom_param = group['custom_parameter']
            for p in group['params']:
                if p.grad is None:
                    continue
                gradTemp=p.grad.data.clone()
                lr=custom_param
                converged=False
                for i in range(self.max_iters):
                    p.requires_grad=False
                    p.data.add_(gradTemp, alpha=-lr)
                    lossNew = closure()
                    armijoTerm=loss.data + torch.sum(gradTemp ** 2) * lr * self.c
                    if lossNew>armijoTerm: #armijo if its inequality, see Numerical Optimization Nocedal
                        p.data.add_(gradTemp, alpha=lr)
                        lr=lr*self.alpha

                    else:
                        converged=True
                        break
                if converged:
                    p.requires_grad=True
                else:

                    p.requires_grad=True
        return loss.data


