import torch

def dot(a: torch.Tensor, b: torch.Tensor):
    """
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    return (a * b).sum(dim=-1)

class L1:
    def __call__(self,batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()

class FLOPS_SIG:
    def __call__(self,batch_rep):
        batch_rep = batch_rep*1000 - 8 #0.01 -> 0.88, 0-> 0.00033
        return torch.sum(torch.mean(torch.sigmoid(batch_rep), dim=0) ** 2)

class L1_SIG:
    def __call__(self,batch_rep):
        batch_rep = batch_rep*1000 - 8
        return torch.sum(torch.sigmoid(batch_rep), dim=-1).mean()

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self,batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return min(self.lambda_,self.lambda_t)