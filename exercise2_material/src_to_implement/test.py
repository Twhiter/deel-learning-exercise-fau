import math
import random

import numpy as np
from scipy.signal import convolve,correlate,convolve2d,correlate2d

from Layers.Conv import Conv


def padding(x,f):
    kernel_width = f.shape[1]
    kernel_height = f.shape[0]


    w1 = math.ceil((kernel_width - 1) / 2)
    w2 = math.floor((kernel_width - 1) / 2)

    h1 = math.ceil((kernel_height - 1) / 2)
    h2 = math.floor((kernel_height - 1) / 2)


    padded_x = np.zeros(np.array(x.shape) + (h1 + h2,w1 + w2))

    # put the image
    padded_x[h1:h1 + x.shape[0], w1:w1 + x.shape[1]] = x

    return padded_x




for t in range(100):

    x = random.randint(20,100)
    y = random.randint(20,100)

    x1 = random.randint(5,10)
    y1 = random.randint(5,10)

    a1 = np.random.randint(1,100,(x,))
    f = np.random.randint(1,100,(x1,))

    c = Conv((1,1),(1,1),1)

    # g = padding(a1,f)



    assert np.sum(correlate(c.unpad1D(c.padding1D(a1,*f.shape),*f.shape),f,mode='same') - correlate(a1,f,mode='same')) < 1e-2,"No." + str(t)

