import numpy as np

import torchvision
import torch


# For any overrides necessary
class EMNIST(torchvision.datasets.EMNIST):
    def __init__(self, *args, **kwargs):
        # self.generated = generated
        # TODO: load generated dataset

        super().__init__(*args, **kwargs)

        self.data = self.data.reshape(self.data.shape[0], -1)
        # Binarize the data
        self.data = np.random.binomial(1, np.true_divide(self.data, self.data.max()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img.astype(np.float32), target

    def __setitem__(self, key, value):
        self.data[key] = value


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)
