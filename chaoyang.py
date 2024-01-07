import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path
from os.path import join
import json

from dataset.utils import noisify


class ChaoYang(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 train=0,
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=1,
                 redux=None,
                 image_size=None
                 ):
        self.image_folder = join(root, 'train')
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.device = device  # 0: hardware; 1: RAM
        self.noise_type = noise_type
        self.random_state = 0
        self.ids = os.listdir(self.image_folder)

        if self.train == 0:  # train
            self.data_list = list(set(self.ids) - set(self.ids[::10]))
        elif self.train == 1:  # test
            self.image_folder = join(root, 'test')
            self.data_list = os.listdir(self.image_folder)
        else:  # val
            self.data_list = self.ids[::10]

        if redux:
            self.data_list = self.data_list[:redux]

        if image_size == None:
            self.imageTransform = transforms.Compose([
                transforms.Resize((512, 512), interpolation=Image.NEAREST)
            ])
        else:
            self.imageTransform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)
            ])

        print("Loading data from {}".format(self.image_folder))
        # now load the picked numpy arrays
        self.data = []
        self.labels = []
        for f in self.data_list:
            label_ = int(f[-5]) 
            data_ = join(self.image_folder, f)
            # print(data_)
            assert os.path.isfile(data_)
            if self.device == 1:
                data_ = self.img_loader(data_)
            self.data.append(data_)
            self.labels.append(label_)

        if self.device == 1:
            self.data == np.concatenate(self.data)

        # noisy labels
        self.labels = np.asarray(self.labels)
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.bool)
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="chaoyang",
                                                                nb_classes=4,
                                                                train_labels=np.expand_dims(self.labels, 1),
                                                                noise_type=noise_type, noise_rate=noise_rate,
                                                                random_state=self.random_state)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def img_loader(self, img_path):
        return np.asarray(self.imageTransform(Image.open(img_path))).astype(np.uint8)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target，index) where target is index of the target class.
        """
        img = self.img_loader(self.data[index]) if self.device == 0 else self.data[index]
        target = self.labels[index] if self.noise_type == 'clean' else self.noisy_labels[index]

        # doing this so that it is consistent with all other datasets
		# to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, torch.from_numpy(np.array(target, dtype=np.int8)).long(), index

    def __len__(self):
        return len(self.data_list)
