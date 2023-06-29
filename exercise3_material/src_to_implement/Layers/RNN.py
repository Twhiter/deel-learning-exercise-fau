from copy import deepcopy

import numpy as np

from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.Sigmoid import Sigmoid
from Layers.SoftMax import SoftMax
from Layers.TanH import TanH


class RNN(BaseLayer):


    def __init__(self,input_size, hidden_size, output_size,memorize=False):
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.memorize = memorize
        self.last_hidden = np.zeros((1,self.hidden_size))

        self.f1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.tanh = TanH()
        self.f2 = FullyConnected(self.hidden_size, self.output_size)
        self.sigmoid = Sigmoid()

        self.gradient_weights = None


    @property
    def optimizer(self):
        return self.f1.optimizer

    @optimizer.setter
    def optimizer(self,optimizer):
        self.f1.optimizer = optimizer
        self.f2.optimizer = deepcopy(optimizer)

    @property
    def weights(self):
        return self.f1.weights

    @weights.setter
    def weights(self,weights):
        self.f1.weights = weights

    def initialize(self, weights_initializer, bias_initializer):

        self.f1.initialize(weights_initializer,bias_initializer)
        self.f2.initialize(weights_initializer,bias_initializer)


    def forward(self,input_tensor):

        self.x_tildes = []
        self.us = []
        self.os = []
        self.hs = []
        self.ys = np.zeros((input_tensor.shape[0],self.output_size))


        if not self.memorize:
            self.last_hidden = np.zeros((1,self.hidden_size))

        for t,xt in enumerate(input_tensor):

            xt = np.expand_dims(xt,axis=0)
            x_tilde = np.hstack((xt,self.last_hidden))
            self.x_tildes.append(x_tilde)

            u = self.f1.forward(x_tilde)
            self.us.append(u)

            h = self.tanh.forward(u)
            self.hs.append(h)

            o = self.f2.forward(h)
            self.os.append(o)

            y = self.sigmoid.forward(o)
            self.ys[t] = y

            self.last_hidden = h

        return self.ys

    def backward(self,error_tensor):

        self.gradient_weights = 0
        accumulated = 0
        error_tensor_to_previous = np.zeros((error_tensor.shape[0],self.input_size))

        for t,et in reversed(list(enumerate(error_tensor))):

            self.sigmoid.forward(self.os[t])
            ot_gradient = self.sigmoid.backward(et)

            self.f2.input_tensor = np.hstack((self.hs[t],np.ones((1,1))))
            ht_gradient = self.f2.backward(ot_gradient) + accumulated

            self.tanh.forward(self.us[t])
            ut_gradient = self.tanh.backward(ht_gradient)

            self.f1.input_tensor = np.hstack((self.x_tildes[t],np.ones((1,1))))
            x_tilde_gradient = self.f1.backward(ut_gradient)

            self.gradient_weights += self.f1.gradient_weights

            error_tensor_to_previous[t] = x_tilde_gradient[0,:self.input_size]
            accumulated = x_tilde_gradient[:,self.input_size:]

        return error_tensor_to_previous