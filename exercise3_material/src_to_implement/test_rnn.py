import unittest

import numpy as np

from Layers.RNN import RNN
from NeuralNetworkTests import L2Loss


class TestRNN(unittest.TestCase):
    def setUp(self):
        self.loss = L2Loss()
        self.rnn = RNN(13, 7, 5)
        self.input_tensor = np.random.rand(13, 9).T
        self.weights = np.random.rand(21, 7)
        self.rnn.weights = self.weights

        self.label_tensor = np.zeros([5, 9]).T
        for i in range(9):
            self.label_tensor[i, np.random.randint(0, 5)] = 1
    
    def compute_loss(self, input_tensor):

        output = self.rnn.forward(input_tensor)
        loss = self.loss.forward(output, self.label_tensor)

        return loss

    def compute_error_and_gradient_weigths(self):
        
        output = self.rnn.forward(self.input_tensor)
        _ = self.loss.forward(output, self.label_tensor)
        error_tensor = self.loss.backward(self.label_tensor)
        error_tensor = self.rnn.backward(error_tensor)
        gradient_weights = self.rnn.gradient_weights

        

        return error_tensor, gradient_weights
    
    def compute_numerical_error(self):
        h = 1e-7
        numerical_error = np.zeros((9, 13))

        loss_true = self.compute_loss(self.input_tensor)

        for i in range(9):
            for j in range(13):
                input_tensor = self.input_tensor.copy()
                input_tensor[i, j] += h
                loss = self.compute_loss(input_tensor)
                numerical_error[i, j] = (loss - loss_true) / h


        return numerical_error

    def compute_numerical_gradient(self):
        h = 1e-7
        numerical_gradient = np.zeros_like(self.rnn.weights)

        loss_true = self.compute_loss(self.input_tensor)

        for i in range(21):
            for j in range(7):
                weights = self.weights.copy()
                weights[i, j] += h

                self.rnn.weights = weights

                loss = self.compute_loss(self.input_tensor)
                numerical_gradient[i, j] = (loss - loss_true) / h

        return numerical_gradient

    
    def test_error_tensor_gradient(self):
        numerical_error = self.compute_numerical_error()
        numerical_gradient = self.compute_numerical_gradient()
        error_tensor, gradient_weights = self.compute_error_and_gradient_weigths()


        print('- numerical_error \n', numerical_error, '\n\n')
        print('- error_tensor \n', error_tensor, '\n\n')

        print('- difference \n', numerical_error - error_tensor, '\n\n')

        np.testing.assert_almost_equal(error_tensor, numerical_error, decimal=4)

        print('-- gradient weight\n',gradient_weights,'\n\n')
        print('--numerical gradient\n',numerical_gradient,'\n\n')