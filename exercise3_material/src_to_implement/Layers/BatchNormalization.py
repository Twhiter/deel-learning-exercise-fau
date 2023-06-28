from copy import deepcopy

import numpy as np

from Layers import Helpers
from Layers.Base import BaseLayer

ε = 1e-10
α = 0.8


class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.bias = None
        self.weights = None


        self.gradient_weights = 0
        self.gradient_bias = 0
        self.trainable = True
        self.channels = channels

        self.initialize(None,None)

        self.mu = None
        self.sigma_square = None
        self.input_tensor = None

        self._optimizer = None
        self._optimizer1 = None



    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,optimizer):
        self._optimizer = optimizer
        self._optimizer1 = deepcopy(optimizer)



    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    def forward_conv(self,input_tensor):

        reformatted_input_tensor = self.reformat(input_tensor)
        out = self.forward_(reformatted_input_tensor)
        return self.reformat(out)


    def forward_(self,input_tensor):
        if not self.testing_phase:

            if self.sigma_square is None and self.mu is None:
                self.sigma_square = np.var(input_tensor, axis=0)
                self.mu = np.mean(input_tensor, axis=0)
            else:
                self.sigma_square = α * self.sigma_square + (1 - α) * np.var(input_tensor, axis=0)
                self.mu = α * self.mu + (1 - α) * np.mean(input_tensor, axis=0)

        if self.testing_phase:
            return self.weights * (input_tensor - self.mu) / np.sqrt(self.sigma_square + ε) + self.bias
        else:
            return self.weights * (input_tensor - np.mean(input_tensor, axis=0)) / np.sqrt(np.var(input_tensor, axis=0) + ε) + self.bias


    def forward(self, input_tensor):
        self.input_tensor = input_tensor.copy()

        if len(self.input_tensor.shape) == 2:
            return self.forward_(input_tensor)
        else:
            return self.forward_conv(input_tensor)

    def backward(self, error_tensor):


        if len(self.input_tensor.shape) > 2:
            formatted_tensor = self.reformat(self.input_tensor)
            formatted_error_tensor = self.reformat(error_tensor)
        else:
            formatted_tensor = self.input_tensor.copy()
            formatted_error_tensor = error_tensor.copy()


        output_tensor = Helpers.compute_bn_gradients(formatted_error_tensor, formatted_tensor, self.weights, self.mu,
                                                     self.sigma_square)


        normalized_tensor = (formatted_tensor - np.mean(formatted_tensor, axis=0)) / np.sqrt(np.var(formatted_tensor, axis=0) + ε)

        self.gradient_weights = np.sum(formatted_error_tensor * normalized_tensor,axis=0)
        self.gradient_bias = np.sum(formatted_error_tensor,axis=0)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._optimizer1 is not None:
            self.bias = self._optimizer1.calculate_update(self.bias,self.gradient_bias)

        if len(self.input_tensor.shape) > 2:
            return self.reformat(output_tensor)
        else:
            return output_tensor

    def reformat(self,tensor):

        # compress that into vector
        if len(tensor.shape) > 2:

            reformatted_tensor = tensor.reshape((*tensor.shape[:2],-1))
            reformatted_tensor = np.transpose(reformatted_tensor,(0,2,1))

            reformatted_tensor = reformatted_tensor.reshape((-1,reformatted_tensor.shape[-1]))

            return reformatted_tensor

        # recover it to image from vector
        else:

            batch_num,channel_num = self.input_tensor.shape[:2]
            image_size = self.input_tensor.shape[2:]

            reformatted_tensor = tensor.reshape((batch_num,np.product(image_size),channel_num))
            reformatted_tensor = np.transpose(reformatted_tensor,(0,2,1))

            reformatted_tensor = reformatted_tensor.reshape((batch_num,channel_num,*image_size))

            return reformatted_tensor

def test():

    batch = BatchNormalization(3)

    error_tensor = np.array([[1, 2, 3],[3,4,5]])

    batch.input_tensor = np.array([[1, 2, 3],[4,5,6]])

    batch.forward(batch.input_tensor)

    batch.weights = np.array([1,2,3])


    for i in range(3):
        batch.backward(error_tensor)

        print(batch.gradient_weights)

# test()

