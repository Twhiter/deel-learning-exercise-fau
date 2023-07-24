import pandas
import skimage
import torch
import torchvision as tv
import torchvision.transforms
from skimage.color import gray2rgb
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor, Normalize

train_mean = [0.59685254, 0.59685254, 0.59685254]  # [0.5783, 0.5783, 0.5783]

train_std = [0.16043035, 0.16043035, 0.16043035]  # [0.1863, 0.1863, 0.1863]


class ChallengeDataset(Dataset):

    def __init__(self, data: pandas.DataFrame, mode: str):
        self.data = data.to_numpy()
        self.mode = mode
        self._transform = tv.transforms.Compose([
            ToPILImage(),
            ToTensor(),
            Normalize(train_mean, train_std),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomErasing(),
            torchvision.transforms
            .RandomApply(transforms=[torchvision.transforms.ColorJitter(.5, .5)], p=.5),
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        filename = self.data[index, 0]

        gray_image = skimage.io.imread(filename, as_gray=True)
        rgb_image = skimage.color.gray2rgb(gray_image)
        rgb_image = skimage.util.img_as_ubyte(rgb_image)

        image = self._transform(rgb_image)

        return image, torch.tensor((self.data[index, 1], self.data[index, 2]), dtype=torch.double)
