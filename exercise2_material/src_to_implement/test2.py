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

        #down sampling
        unsampled_error_tensors = np.zeros((len(self.input_tensor),self.num_kernels,*self.input_tensor[0,0].shape))

        if len(self.stride_shape) == 2:
            unsampled_error_tensors[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensors
        else:
            unsampled_error_tensors[:,:,::self.stride_shape[0]] = error_tensors

        if len(self.convolution_shape) == 3:
            return self.twoD_backward(unsampled_error_tensors)
        else:
            return self.oneD_backward(unsampled_error_tensors)



    def oneD_backward(self,error_tensors):

        kernels = self.weights.copy()

        error_gradients = np.zeros((len(error_tensors), *self.input_tensor[0].shape))
        kernels_gradients = np.zeros((len(error_tensors), self.num_kernels, *self.convolution_shape))

        line_width = self.input_tensor[0, 0].shape[0]

        kernel_width = self.convolution_shape[1]
        padded_width = math.ceil((kernel_width - 1) / 2)

        padded_input_tensor = np.zeros(
            (self.input_tensor.shape[0], self.input_tensor.shape[1],self.input_tensor.shape[2] + 2 * padded_width))

        window_y = np.array(range(0, line_width))

        #put the line
        padded_input_tensor[:, :, padded_width:padded_width + line_width] = self.input_tensor

        for n,sample in enumerate(padded_input_tensor):

            # iterate the channel
            for channel,padded_line in enumerate(sample):

                # iterate correlation window
                for y_idx, y in enumerate(window_y):

                    window = padded_line[y:y + kernel_width]

                    # shift inside the window
                    for shift_y in range(len(window.shape)):
                        p_y = y + shift_y - padded_width

                        # out of line,invalid
                        if p_y < 0 or p_y >= line_width:
                            continue


                        for kernel_idx in range(self.num_kernels):

                            kernels_gradients[n,kernel_idx,shift_y] += error_tensors[n,kernel_idx,y] * padded_line[y + shift_y]

                            error_gradients[n,channel,p_y] += error_tensors[n,kernel_idx,y] * kernels[kernel_idx,channel,shift_y]

        bias_gradients = np.sum(error_tensors, axis=(0, 2))
        self.gradient_weights = np.sum(kernels_gradients, axis=0)
        self.gradient_bias = bias_gradients

        self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)
        self.bias = self._optimizer2.calculate_update(self.bias,self.gradient_bias)



        return error_gradients


    def twoD_backward(self,error_tensors):

        kernels = self.weights.copy()

        error_gradients = np.zeros((len(error_tensors), *self.input_tensor[0].shape))
        kernels_gradients = np.zeros((len(error_tensors), self.num_kernels, *self.convolution_shape))

        image_height,image_width = self.input_tensor[0,0].shape

        kernel_width = self.convolution_shape[2]
        kernel_height = self.convolution_shape[1]

        # w1 = math.ceil((kernel_width - 1) / 2)
        # w2 = math.floor((kernel_width - 1) / 2)
        #
        # h1 = math.ceil((kernel_height - 1) / 2)
        # h2 = math.floor((kernel_height - 1) / 2)
        #
        # padded_x = np.zeros(np.array(x.shape) + (h1 + h2, w1 + w2))

        # calculate the width and height on both sides
        padded_width_left = math.ceil((kernel_width - 1) / 2)
        padded_width_right = math.floor((kernel_width - 1) / 2)

        padded_height_up = math.ceil((kernel_height - 1) / 2)
        padded_height_down = math.floor((kernel_height - 1) / 2)

        padded_input_tensor = np.zeros((self.input_tensor.shape[0],self.input_tensor.shape[1],
                                        self.input_tensor.shape[2] + padded_height_down + padded_height_up,
                                        self.input_tensor.shape[3] + padded_width_right + padded_width_left))
        # put the image
        padded_input_tensor[:,:,padded_height_up:padded_height_up + image_height,padded_width_left:padded_width_left + image_width] \
            = self.input_tensor

        # the kernel window coordinates range over the image
        window_y = np.array(range(0,image_height))
        window_x = np.array(range(0,image_width))

        #iterate the batch
        for n,sample in enumerate(padded_input_tensor):

            # iterate the channel
            for channel,padded_image in enumerate(sample):

                # iterate all the correlation windows
                for y_idx,y in enumerate(window_y):
                    for x_idx,x in enumerate(window_x):

                        window = padded_image[y:y + kernel_height,x:x + kernel_width]

                        # shift inside the window
                        for shift_y,shift_x in np.ndindex(window.shape):

                            p_x = x + shift_x - padded_width_left
                            p_y = y + shift_y - padded_height_up

                            # out of image,invalid
                            if p_x < 0 or p_x >= image_width or p_y < 0 or p_y >= image_height:
                                continue

                            for kernel_idx in range(self.num_kernels):

                                kernels_gradients[n,kernel_idx,channel,shift_y,shift_x] += error_tensors[n,kernel_idx,y,x] \
                                    * padded_image[y + shift_y,x + shift_x]

                                error_gradients[n,channel,p_y,p_x] += error_tensors[n,kernel_idx,y,x]\
                                    * kernels[kernel_idx,channel,shift_y,shift_x]

                                # if p_x == 2 and p_y == 1:
                                #     print(y)
                                #     print(x)
                                #
                                #     print(shift_y)
                                #     print(shift_x)
                                #
                                #
                                #
                                #     print(error_tensors[n,kernel_idx,y,x] * kernels[kernel_idx,channel,shift_y,shift_x])




        bias_gradients = np.sum(error_tensors,axis=(0,2,3))

        self.gradient_weights = np.sum(kernels_gradients,axis=0)
        self.gradient_bias = bias_gradients

        self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        self.bias = self._optimizer2.calculate_update(self.bias, self.gradient_bias)

        return error_gradients

