import math

import numpy as np

from Layers.Base import BaseLayer


class SoftMax(BaseLayer):


    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.y_hat = None

    def forward(self,input_tensor):

        self.input_tensor = input_tensor.copy()

        maxs = np.max(input_tensor, axis=1)
        maxs = np.expand_dims(maxs,axis=-1)

        output = input_tensor - maxs
        output = np.exp(output)
        output = np.sum(output,axis=1)
        output = np.expand_dims(output,axis=-1)

        output = (np.exp(input_tensor - maxs)) / output

        self.y_hat = output.copy()
        return output

    def backward(self,error_tensor):


        return self.y_hat * (error_tensor - np.expand_dims(np.sum(error_tensor * self.y_hat,axis=1),axis=-1))





