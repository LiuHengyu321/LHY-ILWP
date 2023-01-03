import collections
import glob
import logging
import math
import os
import warnings
import torchvision
import numpy as np
from torchvision import datasets, transforms
from typing import Any, Callable, Optional, Tuple
logger = logging.getLogger(__name__)
import pickle
from PIL import Image

class My_dataset(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(My_dataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.fine_targets = []
        self.coarse_targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.fine_targets.extend(entry['fine_labels'])
                self.coarse_targets.extend(entry['coarse_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, fine_targets, coarse_targets = self.data[index], self.fine_targets[index], self.coarse_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, fine_targets, coarse_targets

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)


class iCIFAR100(iCIFAR10):
    base_dataset = My_dataset
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    class_order = [  # Taken from original iCaRL implementation:
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]
    # coarse_class_order = [5, 4, 17, 18, 15, 1, 9, 8, 11, 6, 2, 3, 0, 10, 7, 13, 14, 19, 12, 16]
    """
    [4, 30, 55, 72, 95]
    [1, 32, 67, 73, 91]
    [54, 62, 70, 82, 92]
    [9, 10, 16, 28, 61]
    [0, 51, 53, 57, 83]
    [22, 39, 40, 86, 87]
    [5, 20, 25, 84, 94]
    [6, 7, 14, 18, 24]
    [3, 42, 43, 88, 97]
    [12, 17, 37, 68, 76]
    [23, 33, 49, 60, 71]
    [15, 19, 21, 31, 38]
    [34, 63, 64, 66, 75]
    [26, 45, 77, 79, 99]
    [2, 11, 35, 46, 98]
    [27, 29, 44, 78, 93]
    [36, 50, 65, 74, 80]
    [47, 52, 56, 59, 96]
    [8, 13, 48, 58, 90]
    [41, 69, 81, 85, 89]

    """
    # class_order = [
    #     4, 30, 55, 72, 95,
    #     1, 32, 67, 73, 91,
    #     54, 62, 70, 82, 92,
    #     9, 10, 16, 28, 61,
    #     0, 51, 53, 57, 83,
    #     22, 39, 40, 86, 87,
    #     5, 20, 25, 84, 94,
    #     6, 7, 14, 18, 24,
    #     3, 42, 43, 88, 97,
    #     12, 17, 37, 68, 76,
    #     23, 33, 49, 60, 71,
    #     15, 19, 21, 31, 38,
    #     34, 63, 64, 66, 75,
    #     26, 45, 77, 79, 99,
    #     2, 11, 35, 46, 98,
    #     27, 29, 44, 78, 93,
    #     36, 50, 65, 74, 80,
    #     47, 52, 56, 59, 96,
    #     8, 13, 48, 58, 90,
    #     41, 69, 81, 85, 89,
    # ]
    # class_order = list(np.arange(100))
    trans1 = [
        4, 30, 55, 72, 95,
        1, 32, 67, 73, 91,
        54, 62, 70, 82, 92,
        9, 10, 16, 28, 61,
        0, 51, 53, 57, 83,
        22, 39, 40, 86, 87,
        5, 20, 25, 84, 94,
        6, 7, 14, 18, 24,
        3, 42, 43, 88, 97,
        12, 17, 37, 68, 76,
        23, 33, 49, 60, 71,
        15, 19, 21, 31, 38,
        34, 63, 64, 66, 75,
        26, 45, 77, 79, 99,
        2, 11, 35, 46, 98,
        27, 29, 44, 78, 93,
        36, 50, 65, 74, 80,
        47, 52, 56, 59, 96,
        8, 13, 48, 58, 90,
        41, 69, 81, 85, 89,
    ]

    coarse_class_order = []
    for i in range(len(class_order)):
        if trans1.index(class_order[i]) // 5 not in coarse_class_order:
            coarse_class_order.append(trans1.index(class_order[i]) // 5)

    trans2 = np.zeros(100, dtype=int)
    for i in range(100):
        trans2[trans1[i]] = i // 5
    fine_label_to_coarse = trans2.tolist()

    trans3 = np.zeros(100)
    for i in range(100):
        trans3[i] = class_order.index(trans1[i])
    trans3 = trans3.reshape((20, 5))
    trans4 = np.zeros((20, 5))
    for i in range(20):
        trans4[coarse_class_order.index(i)] = trans3[i]
    trans4 = np.array(trans4, dtype=np.int)
    H1 = np.zeros((20, 100))
    for i in range(20):
        for j in range(5):
            H1[i][trans4[i][j]] = 1
    H = np.array(H1)

