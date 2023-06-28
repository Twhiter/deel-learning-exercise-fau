from Layers.Base import BaseLayer


class Flatten(BaseLayer):


    def __init__(self):
        super().__init__()
        self.shapes = None

    def forward(self,input_tensor):
        self.shapes = input_tensor.shape

        return input_tensor.flatten().reshape((self.shapes[0],-1))

    def backward(self,error_tensor):
        return error_tensor.reshape(self.shapes)

