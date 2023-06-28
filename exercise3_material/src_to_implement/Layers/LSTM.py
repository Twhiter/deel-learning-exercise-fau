import numpy as np

from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.TanH import TanH


class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):


        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.last_hidden = np.zeros(self.hidden_size)
        self.last_cell_state = np.zeros(self.hidden_size)

        self.memorize = False

        s1 = Sigmoid()
        s2 = Sigmoid()
        s3 = Sigmoid()
        s4 = Sigmoid()


        mul1 = Multi()
        mul2 = Multi()
        mul3 = Multi()

        tanH1 = TanH()
        tanH2 = TanH()

        f1 = FullyConnected(self.input_size + self.hidden_size,self.hidden_size)
        f2 = FullyConnected(self.input_size + self.hidden_size,self.hidden_size)
        f3 = FullyConnected(self.input_size + self.hidden_size,self.hidden_size)
        f3 = FullyConnected(self.input_size + self.hidden_size,self.hidden_size)


    def forward(self,input_tensor):

        if not self.memorize:
            self.last_hidden = np.zeros(self.hidden_size)
            self.last_cell_state = np.zeros(self.hidden_size)

class Multi(BaseLayer):


    def __init__(self):
        super().__init__()


    def forward(self,in1,in2):
        self.in1 = in1
        self.in2 = in2
        return in1 * in2

    def backward(self,error_tensor):
        return error_tensor * self.in1,error_tensor * self.in2











