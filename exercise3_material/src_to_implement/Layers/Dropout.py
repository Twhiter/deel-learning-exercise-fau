import numpy as np

from Layers.Base import BaseLayer


class Dropout(BaseLayer):

    def __init__(self,probability):
        super().__init__()
        self.probability = probability


    def forward(self,input_tensor):

        if self.testing_phase:
            return input_tensor

        probs = np.random.uniform(0,1,size=input_tensor.shape)

        self.remove_idx = probs >= self.probability

        output_tensor = input_tensor.copy()
        output_tensor[self.remove_idx] = 0

        return output_tensor / self.probability



    def backward(self,error_tensor):

        error_tensor_previous = error_tensor.copy()
        error_tensor_previous[self.remove_idx] = 0

        return error_tensor_previous / self.probability











