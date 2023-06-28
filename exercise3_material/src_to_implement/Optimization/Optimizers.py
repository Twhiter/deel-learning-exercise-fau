import numpy as np

epsilon = 1e-8


class Optimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularizer_gradient = 0

        return weight_tensor - self.learning_rate * gradient_tensor - regularizer_gradient * self.learning_rate


class SgdWithMomentum(Optimizer):

    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularizer_gradient = 0


        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v - self.learning_rate * regularizer_gradient


class Adam(Optimizer):

    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        # v,r are vectors but initialized as 0
        # will be assigned in calculate_update method
        self.v = 0
        self.r = 0

        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor * gradient_tensor

        v_ = self.v / (1 - pow(self.mu, self.k))
        r_ = self.r / (1 - pow(self.rho, self.k))

        self.k += 1

        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
        else:
            regularizer_gradient = 0

        return weight_tensor - self.learning_rate * v_ / (np.sqrt(r_) + epsilon) - self.learning_rate * regularizer_gradient
