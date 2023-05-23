import abc
import numpy as np
import matplotlib.pyplot as plt


# base class, import package abc.ABC for the convenience of inheritance
class Pattern(abc.ABC):
    @abc.abstractmethod
    def draw(self): pass

    @abc.abstractmethod
    def show(self): pass


class Checker(Pattern):

    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("The resolution should be evenly divided by 2 * tile_size")

        self.tile_size = tile_size
        self.resolution = resolution

        black, white = np.zeros((tile_size, tile_size)), np.ones((tile_size, tile_size))

        first_two = np.hstack((black, white))
        second_two = np.hstack((white, black))

        # constructing the simplest matrix
        self.output = np.vstack((first_two, second_two))

        return

    def draw(self):
        num_tile = self.resolution // self.tile_size
        self.output = np.tile(self.output, (num_tile // 2, num_tile // 2))
        return np.copy(self.output)

    def show(self):
        self.draw()
        plt.imshow(self.output, cmap="gray")
        plt.show()


class Circle(Pattern):

    def __init__(self, resolution, radius, position):
        self.position = position
        self.radius = radius
        self.resolution = resolution
        self.output = np.zeros((resolution, resolution))
        return

    def draw(self):
        x_count = np.arange(0, self.resolution, 1)
        x, y = np.meshgrid(x_count, x_count)

        # get the indices of the elements that are within the circle
        indices = ((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2) <= self.radius ** 2

        self.output[indices] = 1

        return self.output.copy()

    def show(self):
        self.draw()

        plt.imshow(self.output, cmap="gray")
        plt.show()


class Spectrum(Pattern):

    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
        return

    def draw(self):
        # make rgb array based on column or row
        r = np.linspace(0, 1, self.resolution)
        g = np.linspace(0, 1, self.resolution)
        b = np.flip(np.linspace(0, 1, self.resolution))

        g = np.tile(g, (self.resolution, 1)).T
        r = np.tile(r, (self.resolution, 1))
        b = np.tile(b, (self.resolution, 1))

        self.output = np.stack((r, g, b), axis=2)

        return self.output.copy()

    def show(self):
        self.draw()
        plt.imshow(self.output)
        plt.show()
