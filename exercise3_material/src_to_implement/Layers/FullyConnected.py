from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self,input_size,output_size):
        super().__init__()

        self.trainable = True
        self._optimizer = None
        self.weights = np.random.uniform(low=0,high=1,size=(input_size + 1,output_size))

        self.input_tensor = None
        self.gradient_weights = None

        self.input_size = input_size
        self.output_size = output_size


    def forward(self,input_tensor):

        bias = np.ones((input_tensor.shape[0],1))
        input_tensor = np.hstack((input_tensor,bias))

        self.input_tensor = input_tensor.copy()

        return input_tensor @ self.weights

    def backward(self,error_tensor):
        e_n_1 = error_tensor @ self.weights.T[:,:-1]

        self.gradient_weights = self.input_tensor.T @ error_tensor

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient_weights)
        return e_n_1



    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,optimizer):
        self._optimizer = optimizer

    def initialize(self,weights_initializer,bias_initializer):

        weights = weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        bias = bias_initializer.initialize((1,self.output_size),self.input_size,self.output_size)

        self.weights = np.vstack((weights,bias))










