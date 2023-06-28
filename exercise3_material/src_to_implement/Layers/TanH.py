import numpy as np

from Layers.Base import BaseLayer


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 2 / (1 + np.exp(-2 * input_tensor)) - 1

        return self.activation

    def backward(self, error_tensor):
        return error_tensor * (1 - self.activation * self.activation)
