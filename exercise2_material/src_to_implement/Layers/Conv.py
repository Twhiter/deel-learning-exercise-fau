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
        self._optimizer = None
        self._optimizer2 = None

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
        batch_num = len(self.input_tensor)

        # down sampling
        upsampled_error_tensors = np.zeros((batch_num, self.num_kernels, *self.input_tensor[0, 0].shape))

        if len(self.stride_shape) == 2:
            upsampled_error_tensors[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensors
            pad = self.padding2D
            unpad = self.unpad2D
        else:
            upsampled_error_tensors[:, :, ::self.stride_shape[0]] = error_tensors
            pad = self.padding1D
            unpad = self.unpad1D

        kernel_gradients = np.zeros((batch_num, self.num_kernels, *self.convolution_shape))
        error_gradients = np.zeros((batch_num, *self.input_tensor[0].shape))
        bias_gradients = np.zeros((batch_num, self.num_kernels))

        for n, error_tensor in enumerate(upsampled_error_tensors):
            sample = self.input_tensor[n]

            for kernel_idx in range(self.num_kernels):

                for channel, image in enumerate(sample):
                    padded_image = pad(image, *self.convolution_shape[1:])
                    kernel_gradients[n, kernel_idx, channel] = correlate(padded_image, error_tensor[kernel_idx],
                                                                         mode='valid')

                    t = convolve(error_tensor[kernel_idx], self.weights[kernel_idx, channel], mode='full')
                    error_gradients[n, channel] += unpad(t, *self.convolution_shape[1:])

                bias_gradients[n, kernel_idx] = np.sum(error_tensor[kernel_idx])

        self.gradient_weights = np.sum(kernel_gradients, axis=0)
        self.gradient_bias = np.sum(bias_gradients, axis=0)

        if self._optimizer is not None and self._optimizer2 is not None:

            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer2.calculate_update(self.bias, self.gradient_bias)

        return error_gradients



    def padding2D(self, sample, kernel_height, kernel_width):

        w1 = math.ceil((kernel_width - 1) / 2)
        w2 = math.floor((kernel_width - 1) / 2)

        h1 = math.ceil((kernel_height - 1) / 2)
        h2 = math.floor((kernel_height - 1) / 2)

        padded_sample = np.zeros(np.array(sample.shape) + (h1 + h2, w1 + w2))

        padded_sample[h1:h1 + sample.shape[0], w1:w1 + sample.shape[1]] = sample

        return padded_sample

    def unpad2D(self, padded_sample, kernel_height, kernel_width):

        w1 = math.ceil((kernel_width - 1) / 2)
        w2 = math.floor((kernel_width - 1) / 2)

        h1 = math.ceil((kernel_height - 1) / 2)
        h2 = math.floor((kernel_height - 1) / 2)

        return padded_sample[h1:padded_sample.shape[0] - h2,w1:padded_sample.shape[1] - w2]

    def padding1D(self,sample,kernel_width):

        w1 = math.ceil((kernel_width - 1) / 2)
        w2 = math.floor((kernel_width - 1) / 2)

        padded_sample = np.zeros(np.array(sample.shape[0] + w1 + w2))

        padded_sample[w1:w1 + sample.shape[0]] = sample

        return padded_sample

    def unpad1D(self,padded_sample,kernel_width):

        w1 = math.ceil((kernel_width - 1) / 2)
        w2 = math.floor((kernel_width - 1) / 2)

        return padded_sample[w1:padded_sample.shape[0] - w2]





