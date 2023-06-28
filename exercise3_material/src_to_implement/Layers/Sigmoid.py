import numpy as np

from Layers.Base import BaseLayer


class Sigmoid(BaseLayer):



    def __init__(self):
        super().__init__()
        self.activation_function = None

    def forward(self,input_tensor):

        self.activation_function = 1 / (1 + np.exp(-input_tensor))
        return self.activation_function


    def backward(self,error_tensor):
        return error_tensor * self.activation_function * (1 - self.activation_function)