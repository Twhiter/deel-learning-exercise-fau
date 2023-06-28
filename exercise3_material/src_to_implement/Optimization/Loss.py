import numpy
import numpy as np


class CrossEntropyLoss:


    def __init__(self):
        self.prediction_tensor = None

    def forward(self,prediction_tensor, label_tensor):
        slices = label_tensor == 1
        self.prediction_tensor = prediction_tensor.copy()

        loss = np.sum(-numpy.log(prediction_tensor[slices] + np.finfo(np.float64).eps))

        return loss

    def backward(self,label_tensor):

        En = - label_tensor / (self.prediction_tensor + np.finfo(np.float64).eps)
        return En




