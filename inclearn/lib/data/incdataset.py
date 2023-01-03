import logging
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
sys.path.append("lib\\data")
from datasets import iCIFAR10, iCIFAR100


logger = logging.getLogger(__name__)


class IncrementalDataset:
    """Incremental generator of datasets.

    :param dataset_name: Among a list of available dataset, that can easily
                         be defined (see at file's end).
    :param random_order: Shuffle the class ordering, else use a cherry-picked
                         ordering.
    :param shuffle: Shuffle batch order between epochs.
    :param workers: Number of workers loading the data.
    :param batch_size: The batch size.
    :param seed: Seed to force determinist class ordering.
    :param increment: Number of class to add at each task.
    :param validation_split: Percent of training data to allocate for validation.
    :param onehot: Returns targets encoded as onehot vectors instead of scalars.
                   Memory is expected to be already given in an onehot format.
    :param initial_increment: Initial increment may be defined if you want to train
                              on more classes than usual for the first task, like
                              UCIR does.
    """

    def __init__(
        self,
        dataset_name,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.,
        onehot=False,
        initial_increment=None,
        sampler=None,
        sampler_config=None,
        data_path="data",
        class_order=None,
        dataset_transforms=None,
        all_test_classes=False,
        metadata_path=None
    ):
        datasets = _get_datasets(dataset_name)
        print(datasets)
        dataset1 = datasets[0]

        self._setup_data(
            dataset1,
            random_order=random_order,
            class_order=class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            initial_increment=initial_increment,
            data_path=data_path
        )
        dataset = datasets[0]()
        # self.dataset_p = dataset
        dataset.set_custom_transforms(dataset_transforms)
        self.train_transforms = dataset.train_transforms  # FIXME handle multiple datasets
        self.test_transforms = dataset.test_transforms
        self.common_transforms = dataset.common_transforms

        self.open_image = datasets[0].open_image

        self._current_task = 0

        self._seed = seed
        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self._onehot = onehot
        self._sampler = sampler
        self._sampler_config = sampler_config
        self._all_test_classes = all_test_classes
        self.train_fine_nums_inc = 0
        self.train_coarse_nums_inc = []
        self.H = dataset.H
    @property
    def n_tasks(self):
        return len(self.increments)

    @property
    def n_classes(self):
        return sum(self.increments)

    def new_task(self, memory=None, memory_val=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train_fine, y_train_coarse = self._select(
            self.data_train, self.fine_targets_train, self.coarse_targets_train, low_range=min_class, high_range=max_class
        )
        self.train_fine_nums_inc, self.train_coarse_nums_inc = self.get_class_nums(y_train_fine, y_train_coarse)
        nb_new_classes = len(np.unique(y_train_fine))
        x_val, y_val_fine, y_val_coarse = self._select(
            self.data_val, self.fine_targets_val, self.coarse_targets_val, low_range=min_class,
            high_range=max_class
        )
        if self._all_test_classes is True:
            logger.info("Testing on all classes!")
            x_test, y_test_fine, y_test_coarse = self._select(
                self.data_test, self.fine_targets_test, self.coarse_targets_test, high_range=sum(self.increments)
            )
        elif self._all_test_classes is not None or self._all_test_classes is not False:
            max_class = sum(self.increments[:self._current_task + 1 + self._all_test_classes])
            logger.info(
                f"Testing on {self._all_test_classes} unseen tasks (max class = {max_class})."
            )
            x_test, y_test_fine, y_test_coarse = self._select(
                self.data_test, self.fine_targets_test, self.coarse_targets_test, high_range=max_class
            )
        else:
            x_test, y_test_fine, y_test_coarse = self._select(
                self.data_test, self.fine_targets_test, self.coarse_targets_test, high_range=max_class)

        if self._onehot:

            def to_onehot(x):
                n = np.max(x) + 1
                return np.eye(n)[x]

            y_train_fine = to_onehot(y_train_fine)
            y_train_coarse = to_onehot(y_train_coarse)
        if memory is not None:
            logger.info("Set memory of size: {}.".format(memory[0].shape[0]))
            x_train, y_train_fine, y_train_coarse, train_memory_flags = self._add_memory(x_train, y_train_fine,
                                                                                         y_train_coarse, *memory)
        else:
            train_memory_flags = np.zeros((x_train.shape[0],))
        if memory_val is not None:
            logger.info("Set validation memory of size: {}.".format(memory_val[0].shape[0]))
            x_val, y_val_fine, y_val_coarse, val_memory_flags = self._add_memory(x_val, y_val_fine, y_val_coarse,
                                                                                 *memory)
        else:
            val_memory_flags = np.zeros((x_val.shape[0],))
        train_loader = self._get_loader(x_train, y_train_fine, y_train_coarse, train_memory_flags, mode="train")
        val_loader = self._get_loader(x_val, y_val_fine, y_val_coarse, val_memory_flags,
                                      mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test_fine, y_test_coarse, np.zeros((x_test.shape[0],)), mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "total_n_classes": sum(self.increments),
            "increment": nb_new_classes,  # self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0],
            "fine_class_num": self.train_fine_nums_inc,
            "coarse_class_num": self.train_coarse_nums_inc,
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _add_memory(self, x, y_fine, y_coarse, data_memory, fine_targets_memory, coarse_targets_memory):
        if self._onehot:  # Need to add dummy zeros to match the number of targets:
            fine_targets_memory = np.concatenate(
                (
                    fine_targets_memory,
                    np.zeros((fine_targets_memory.shape[0], self.increments[self._current_task]))
                ),
                axis=1
            )
            coarse_targets_memory = np.concatenate(
                (
                    coarse_targets_memory,
                    np.zeros((coarse_targets_memory.shape[0], self.increments[self._current_task]))
                ),
                axis=1
            )
        memory_flags = np.concatenate((np.zeros((x.shape[0],)), np.ones((data_memory.shape[0],))))

        x = np.concatenate((x, data_memory))
        y_fine = np.concatenate((y_fine, fine_targets_memory))
        y_coarse = np.concatenate((y_coarse, coarse_targets_memory))

        return x, y_fine, y_coarse, memory_flags

    def get_custom_loader(
        self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        """Returns a custom loader.

        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y_fine, y_coarse = self.data_train, self.fine_targets_train, self.coarse_targets_train
        elif data_source == "val":
            x, y_fine, y_coarse = self.data_val, self.fine_targets_val, self.coarse_targets_val
        elif data_source == "test":
            x, y_fine, y_coarse = self.data_test, self.fine_targets_test, self.coarse_targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, fine_targets, coarse_targets = [], [], []
        for class_index in class_indexes:
            class_data, fine_class_targets, coarse_class_targets = self._select(
                x, y_fine, y_coarse, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            fine_targets.append(fine_class_targets)
            coarse_targets.append(coarse_class_targets)

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            fine_targets = np.concatenate(fine_targets)
            coarse_targets = np.concatenate(coarse_targets)

        if (not isinstance(memory, tuple) and
            memory is not None) or (isinstance(memory, tuple) and memory[0] is not None):
            if len(data) > 0:
                data, fine_targets, coarse_targets, memory_flags = self._add_memory(data, fine_targets, coarse_targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self._get_loader(
            data, fine_targets, coarse_targets, memory_flags, shuffle=False, mode=mode, sampler=sampler
        )

    def get_memory_loader(self, data, fine_targets, coarse_targets):
        return self._get_loader(
            data, fine_targets, coarse_targets, np.ones((data.shape[0],)), shuffle=True, mode="train"
        )

    def _select(self, x, y_fine, y_coarse, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y_fine >= low_range, y_fine < high_range))[0]
        return x[idxes], y_fine[idxes], y_coarse[idxes]

    def get_class_nums(self, y_fine, y_coarse):
        return len(np.unique(y_fine)), list(np.unique(y_coarse))

    def _get_loader(self, x, y_fine, y_coarse, memory_flags, shuffle=True, mode="train", sampler=None):
        if mode == "train":
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
        elif mode == "test":
            trsf = transforms.Compose([*self.test_transforms, *self.common_transforms])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=1.), *self.test_transforms,
                    *self.common_transforms
                ]
            )
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        sampler = sampler or self._sampler
        if sampler is not None and mode == "train":
            logger.info("Using sampler {}".format(sampler))
            sampler = sampler(y_fine, memory_flags, batch_size=self._batch_size, **self._sampler_config)
            batch_size = 1
        else:
            sampler = None
            batch_size = self._batch_size

        return DataLoader(
            DummyDataset(x, y_fine, y_coarse, memory_flags, trsf, open_image=self.open_image),
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=self._workers,
            batch_sampler=sampler
        )

    def _setup_data(
        self,
        dataset,
        random_order=False,
        class_order=None,
        seed=1,
        increment=10,
        validation_split=0.,
        initial_increment=None,
        data_path="data"
    ):
        # FIXME: handles online loading of images
        self.data_train, self.fine_targets_train, self.coarse_targets_train = [], [], []
        self.data_test, self.fine_targets_test, self.coarse_targets_test = [], [], []
        self.data_val, self.fine_targets_val, self.coarse_targets_val = [], [], []
        self.fine_increments = []
        self.coarse_increments = []
        self.fine_class_order = []
        self.coarse_class_order = []
        self.increments = []

        train_dataset = dataset().base_dataset(data_path, train=True, download=True)
        test_dataset = dataset().base_dataset(data_path, train=False, download=True)

        x_train = train_dataset.data
        y_train_fine, y_train_coarse = np.array(train_dataset.fine_targets), np.array(train_dataset.coarse_targets)

        x_val, y_val_fine, y_val_coarse, x_train, y_train_fine, y_train_coarse = self._split_per_class(
            x_train, y_train_fine, y_train_coarse, validation_split
        )

        x_test = test_dataset.data
        y_test_fine, y_test_coarse = np.array(test_dataset.fine_targets), np.array(test_dataset.coarse_targets)
        fine_order = dataset.class_order
        logger.info("Dataset {}: class ordering: {}.".format(dataset.__name__, fine_order))
        self.fine_class_order = fine_order
        coarse_order = dataset.coarse_class_order
        self.coarse_class_order = coarse_order
        if initial_increment is None:
            nb_steps = len(fine_order) / increment
            remainder = len(fine_order) - int(nb_steps) * increment

            if not nb_steps.is_integer():
                logger.warning(
                    f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                )
                self.increments = [increment for _ in range(int(nb_steps))]
                self.increments.append(remainder)
            else:
                self.increments = [increment for _ in range(int(nb_steps))]
        else:
            self.increments = [initial_increment]

            nb_steps = (len(fine_order) - initial_increment) / increment
            remainder = (len(fine_order) - initial_increment) - int(nb_steps) * increment
            if not nb_steps.is_integer():
                logger.warning(
                    f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                )
                self.increments.extend([increment for _ in range(int(nb_steps))])
                self.increments.append(remainder)
            else:
                self.increments.extend([increment for _ in range(int(nb_steps))])

        y_train_fine = self._map_new_class_index(y_train_fine, fine_order)
        y_val_fine = self._map_new_class_index(y_val_fine, fine_order)
        y_test_fine = self._map_new_class_index(y_test_fine, fine_order)

        y_train_coarse = self._map_new_class_index(y_train_coarse, coarse_order)
        y_val_coarse = self._map_new_class_index(y_val_coarse, coarse_order)
        y_test_coarse = self._map_new_class_index(y_test_coarse, coarse_order)

        self.data_train = x_train
        self.fine_targets_train = y_train_fine
        self.coarse_targets_train = y_train_coarse

        self.data_val = x_val
        self.fine_targets_val = y_val_fine
        self.coarse_targets_val = y_val_coarse

        self.data_test = x_test
        self.fine_targets_test = y_test_fine
        self.coarse_targets_test = y_test_coarse

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y_fine, y_coarse, validation_split=0.):
        """Splits train data for a subset of validation data.

        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y_fine = y_fine[shuffled_indexes]
        y_coarse = y_coarse[shuffled_indexes]

        x_val, y_val_fine, y_val_coarse = [], [], []
        x_train, y_train_fine, y_train_coarse = [], [], []

        for class_id in np.unique(y_fine):
            class_indexes = np.where(y_fine == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val_fine.append(y_fine[val_indexes])
            y_val_coarse.append(y_coarse[val_indexes])
            x_train.append(x[train_indexes])
            y_train_fine.append(y_fine[train_indexes])
            y_train_coarse.append(y_coarse[train_indexes])

        x_train = np.concatenate(x_train)
        y_train_fine, y_train_coarse = np.concatenate(y_train_fine), np.concatenate(y_train_coarse)

        x_val = np.concatenate(x_val)
        y_val_fine, y_val_coarse = np.concatenate(y_val_fine), np.concatenate(y_val_coarse)

        return x_val, y_val_fine, y_val_coarse, x_train, y_train_fine, y_train_coarse


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y_fine, y_coarse, memory_flags, trsf, open_image=False):
        self.x, self.y_fine, self.y_coarse = x, y_fine, y_coarse
        self.memory_flags = memory_flags
        self.trsf = trsf
        self.open_image = open_image

        assert x.shape[0] == y_fine.shape[0] == y_coarse.shape[0] == memory_flags.shape[0]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y_fine, y_coarse = self.x[idx], self.y_fine[idx], self.y_coarse[idx]
        memory_flag = self.memory_flags[idx]

        if self.open_image:
            img = Image.open(x).convert("RGB")
        else:
            img = Image.fromarray(x.astype("uint8"))

        img = self.trsf(img)
        return {"inputs": img, "fine_targets": y_fine, "coarse_targets": y_coarse, "memory_flags": memory_flag}


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

