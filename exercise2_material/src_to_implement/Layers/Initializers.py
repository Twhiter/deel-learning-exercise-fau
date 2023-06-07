import math

import numpy as np


class Constant:

    def __init__(self,value=0.1):
        self.value = value


    def initialize(self,weight_shape,fan_in,fan_out):
        tensor = np.zeros(weight_shape)
        tensor[:] = self.value
        return tensor

class UniformRandom:

    def __init__(self):
        pass

    def initialize(self,weight_shape,fan_in,fan_out):
        return np.random.uniform(0,1,weight_shape)




class Xavier:

    def __init__(self):
        pass

    def initialize(self,weight_shape,fan_in,fan_out):

        sigma = math.sqrt(2 / (fan_in + fan_out))

        return np.random.normal(0,sigma,weight_shape)


class He:

    def __init__(self):
        pass

    def initialize(self,weight_shape,fan_in,fan_out):

        sigma = math.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weight_shape)


