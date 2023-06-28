import numpy as np

from Layers.Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()

        self.input_tensors = None
        self.maxargs = None

        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensors):

        window_x = np.array(range(input_tensors[0, 0].shape[0] - self.pooling_shape[0] + 1))
        window_y = np.array(range(input_tensors[0, 0].shape[1] - self.pooling_shape[1] + 1))

        maxargs = np.empty((len(input_tensors), len(input_tensors[0]), len(window_x), len(window_y)), dtype=object)
        maxs = np.zeros((len(input_tensors), len(input_tensors[0]), len(window_x), len(window_y)))

        for n, sample in enumerate(input_tensors):

            for channel, image in enumerate(sample):
                for x_idx, x in enumerate(window_x):
                    for y_idx, y in enumerate(window_y):
                        window = image[x:x + self.pooling_shape[0], y:y + self.pooling_shape[1]]
                        maxargs[n, channel, x_idx, y_idx] = np.unravel_index(np.argmax(window), window.shape)
                        maxs[n, channel, x_idx, y_idx] = window[maxargs[n, channel, x_idx, y_idx]]

        # down sampling
        # maxargs = maxargs[:,:,0::self.stride_shape[0],0::self.stride_shape[1]]
        maxs = maxs[:, :, 0::self.stride_shape[0], 0::self.stride_shape[1]]

        self.maxargs = maxargs
        self.input_tensors = input_tensors
        return maxs

    def backward(self, error_tensors):

        # up sampling
        upsampled_error_tensors = np.zeros(self.maxargs.shape)
        upsampled_error_tensors[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensors

        error_gradients = np.zeros(self.input_tensors.shape)

        window_x = np.array(range(self.input_tensors[0, 0].shape[0] - self.pooling_shape[0] + 1))
        window_y = np.array(range(self.input_tensors[0, 0].shape[1] - self.pooling_shape[1] + 1))

        for n, sample in enumerate(self.input_tensors):

            for channel, image in enumerate(sample):
                for x_idx, x in enumerate(window_x):
                    for y_idx, y in enumerate(window_y):

                        pixel_x, pixel_y = np.array([x, y]) + self.maxargs[n, channel, x_idx, y_idx]

                        error_gradients[n, channel, pixel_x, pixel_y] += upsampled_error_tensors[n, channel, x_idx, y_idx]

        return error_gradients
