import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a
# next function. This next function returns the next generated object. In our case it returns the input of a neural
# network each time it gets called. This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.shuffle = shuffle
        self.mirroring = mirroring
        self.rotation = rotation
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_path = label_path
        self.file_path = file_path

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        self.epoch = 0
        self.idx = -1
        self.batches = None

        self.loadData()

    # preprocessing and generate batch array
    def loadData(self):
        # open the json file to load the labels
        with open(self.label_path, 'r') as label_file:
            labels = np.array(list(json.load(label_file).items()))
        if self.shuffle:
            np.random.shuffle(labels)

        # generate the batches
        batch_num = int(math.ceil(len(labels) / self.batch_size))
        self.batches = [labels[i * self.batch_size:(i + 1) * self.batch_size] for i in range(0, batch_num)]

        # compensate the last batch by introducing the elements
        # from the beginning of the labels
        if len(self.batches[-1]) < self.batch_size:
            dif = self.batch_size - len(self.batches[-1])
            if self.shuffle:
                np.random.shuffle(labels)
            self.batches[-1] = np.concatenate((self.batches[-1], labels[0:dif]))

        # indicate which batches the iterator is on
        self.idx = -1

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        # if run out of all data set
        if self.idx + 1 >= len(self.batches):
            self.loadData()
            self.epoch = self.epoch + 1

        self.idx = self.idx + 1

        batch = self.batches[self.idx]
        # read images from the directory and label integer from the batch
        images = [np.load(self.file_path + str(imgNum) + ".npy") for imgNum, _ in batch]
        labels = [label for _, label in batch]

        # resize the image and then perform the transformation
        images = np.array([self.augment(resize(img, self.image_size)) for img in images])

        # # get the string of the corresponding int label
        # labels = [self.class_name(label) for label in labels]

        labels = [int(label) for label in labels]

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        # rotate
        if self.rotation:
            rotate_times = random.randint(1, 3)
            img = np.rot90(img, rotate_times)

        # mirroring
        if self.mirroring:
            if random.randint(1, 2) == 1:
                img = np.flipud(img)
            else:
                img = np.fliplr(img)

        return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, int_label):
        # This function returns the class name for a specific input
        return self.class_dict[int(int_label)]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method

        images, labels = self.next()

        # show at most 12 images
        plt.figure()

        for i in range(1, min(len(images), 12) + 1):
            ax = plt.subplot(4, 3, i)
            plt.imshow(images[i - 1])
            ax.set_title(self.class_name(labels[i - 1]))

        plt.show()
