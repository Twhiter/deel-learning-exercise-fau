import numpy as np

epsilon = 1e-8

class Sgd:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self,weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:

    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.v = 0


    def calculate_update(self,weight_tensor, gradient_tensor):

        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v

class Adam:

    def __init__(self,learning_rate,mu,rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        # v,r are vectors but initialized as 0
        # will be assigned in calculate_update method
        self.v = 0
        self.r = 0

        self.k = 1

    def calculate_update(self,weight_tensor, gradient_tensor):

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor

        v_ = self.v / (1 - pow(self.mu,self.k))
        r_ = self.r / (1 - pow(self.rho,self.k))

        self.k += 1

        return weight_tensor - self.learning_rate * v_ / (np.sqrt(r_ ) + epsilon)







