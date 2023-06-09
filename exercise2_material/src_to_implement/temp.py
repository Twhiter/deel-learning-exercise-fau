import copy
import math

import numpy as np
from scipy.signal import convolve,correlate

from Layers.Base import BaseLayer
from Optimization.Optimizers import Sgd


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()

        self.input_tensor = None
        self._optimizer = Sgd(1)
        self._optimizer2 = Sgd(1)

        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0, 1, (num_kernels,) + self.convolution_shape)
        self.bias = np.random.uniform(0, 1, num_kernels)

        self.gradient_weights = 0
        self.gradient_bias = 0

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer2 = copy.deepcopy(self._optimizer)

    def initialize(self, weights_initializer, bias_initializer):

        self.weights = weights_initializer.initialize((self.num_kernels,) + self.convolution_shape, np.prod(self.convolution_shape),
                                                      np.prod((self.num_kernels,) + self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.num_kernels, np.prod(self.convolution_shape),
                                                np.prod((self.num_kernels,) + self.convolution_shape[1:]))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        row = []

        # iterate over the batch
        for sample in input_tensor:
            block = []

            # iterate over the output channel
            for i in range(self.num_kernels):
                v = 0

                # iterate over the input channel
                for j in range(len(sample)):
                    v = v + correlate(sample[j], self.weights[i][j], mode='same')

                dx = self.stride_shape[0]
                if len(self.stride_shape) == 2:
                    dy = self.stride_shape[1]
                    v = v[0::dx, 0::dy] + self.bias[i]
                else:
                    v = v[0::dx]

                block.append(v)

            row.append(block)

        row = np.array(row)
        return row

    def backward(self, error_tensors):

        num_batch = len(error_tensors)
        bias_gradients = np.zeros((num_batch,self.num_kernels))
        error_gradients = np.zeros((num_batch,*self.input_tensor[0].shape))
        kernels_gradients = np.zeros((num_batch,self.num_kernels, *self.convolution_shape))

        # iterate over batch
        for n in range(num_batch):

            sample = self.input_tensor[n]
            error_tensor = error_tensors[n]

            # output channel depth
            for i in range(self.num_kernels):
                # input channel depth
                for j in range(self.convolution_shape[0]):

                    up_sampled_error_tensor = np.zeros(sample[j].shape)
                    if len(self.stride_shape) == 1:
                        up_sampled_error_tensor[0::self.stride_shape[0]] = error_tensor[i]
                    else:
                        up_sampled_error_tensor[0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor[i]

                    slices = np.zeros(up_sampled_error_tensor.shape, dtype='bool')

                    if len(self.convolution_shape[1:]) == 1:
                        kx = math.ceil((self.convolution_shape[1] + 1) / 2) - 1
                        xs = np.array(range(-kx, self.convolution_shape[1] - kx)) + math.ceil((sample[j].shape[0] + 1) / 2) - 1

                        slices[xs] = True
                    else:

                        kx = math.ceil((self.convolution_shape[1] + 1) / 2) - 1
                        ky = math.ceil((self.convolution_shape[2] + 1) / 2) - 1

                        xs = np.array(range(-kx,self.convolution_shape[1] - kx)) + math.ceil((sample[j].shape[0] + 1) / 2) - 1
                        ys = np.array(range(-ky,self.convolution_shape[2] - ky)) + math.ceil((sample[j].shape[1] + 1) / 2) - 1

                        x,y = np.meshgrid(xs,ys)

                        slices[x,y] = True

                    kernels_gradients[n, i, j] = (correlate(sample[j], up_sampled_error_tensor, mode='same')[slices]).reshape(self.convolution_shape[1:])
                    error_gradients[n, j] += convolve(up_sampled_error_tensor, self.weights[i, j], mode='same')

                bias_gradients[n][i] = np.sum(error_tensor[i])

        self.gradient_weights = np.sum(kernels_gradients,axis=0)
        self.gradient_bias = np.sum(bias_gradients,axis=0)

        self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        self.bias = self._optimizer2.calculate_update(self.bias, self.gradient_bias)

        return error_gradients


X = np.array([1,2,3,4,5,6,7,8,9]).reshape((3,3))
L = np.ones(X.shape)
F = np.array([[[[1,2],[3,4]]]])



c = Conv((1,1),(1,2,2),1)
c.forward(np.array([[X]]))
c.weights = F

d = c.backward(np.array([[L]]))

print(c.gradient_weights)
print(d)
