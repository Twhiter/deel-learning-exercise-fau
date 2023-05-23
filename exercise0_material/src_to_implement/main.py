import numpy as np
import matplotlib.pyplot as plt

import pattern
from generator import ImageGenerator
from pattern import Checker, Circle

label_path = './Labels.json'
file_path = './exercise_data/'
if __name__ == '__main__':
    # checker = Checker(250, 25)
    # checker.show()

    # circle = Circle(1000, 20, (50, 50))
    # circle.show()
    #
    # spe = pattern.Spectrum(100)
    #
    # spe.show()

    g1 = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    # generator.show()
    g2 = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)

    g2.show()
    print('')




