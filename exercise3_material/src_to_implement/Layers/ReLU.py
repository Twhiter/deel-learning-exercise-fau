import numpy as np

from Layers.Base import BaseLayer


class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        input_tensor_ = input_tensor.copy()
        self.input_tensor = input_tensor.copy()

        input_tensor_[input_tensor_ < 0] = 0
        return input_tensor_

    def backward(self, error_tensor):
        derivative = self.input_tensor.copy()
        derivative[derivative < 0] = 0
        derivative[derivative > 0] = 1

        return derivative * error_tensor
