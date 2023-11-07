# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from PIL import Image
from albumentations import PadIfNeeded, HorizontalFlip, VerticalFlip, CenterCrop, Crop, Compose, Transpose, RandomRotate90, ShiftScaleRotate
from albumentations import ElasticTransform, GridDistortion, OpticalDistortion, RandomCrop, OneOf, CLAHE, RandomContrast, RandomBrightnessContrast
# import cv2
from torchvision import transforms
from torch.utils.data import Dataset


root = 'semi/data/'
mean, std = 0, 255

def load_image(file):
    return Image.open(file)


def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


class Mars(Dataset):

    earth_list = "lists/earth2.txt"
    mars_list = "lists/mars_train.txt"
    mars_un = "lists/mars.txt"
    test_list = "lists/mars_test.txt"

    def __init__(self, root, train_phase=True, labeled=True, mode='mars', aug=True):
        """
        mars: using mars images
        """
        np.random.seed(1)
        self.n_class = 1
        self.root = root
        self.aug = aug
        self.train_phase = train_phase

        self.images_root = os.path.join(self.root, 'imgs')
        self.labels_root = os.path.join(self.root, 'masks')
        self.earth = read_img_list(os.path.join(self.root, self.earth_list))
        self.mars = read_img_list(os.path.join(self.root, self.mars_list))
        self.mars_un = read_img_list(os.path.join(self.root, self.mars_un))
        self.test = read_img_list(os.path.join(self.root, self.test_list))

        if train_phase:
            self.labeled = labeled
            if mode == 'earth':
                self.img_list = self.earth
            elif mode == 'mars':
                self.img_list = self.mars
            elif mode == 'earth_mars':
                self.img_list = self.mars + self.earth
            elif mode == 'ae':
                self.img_list = self.mars_un
        else:
            self.labeled = True
            self.img_list = self.test

    def transform(self, image, mask):
        aug = Compose([
            HorizontalFlip(p=0.9),
            RandomBrightnessContrast(p=.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20,
                             p=0.7, border_mode=0, interpolation=4),
            RandomCrop(height=224, width=224)
        ])

        augmented = aug(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def __getitem__(self, index):
        filename = self.img_list[index]
        with open(os.path.join(self.images_root, filename+'.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        if self.labeled:
            with open(os.path.join(self.labels_root, filename+'.png'), 'rb') as f:
                label = load_image(f).convert('P')
        else:
            label = 0
        image = (np.array(image).astype(np.float32) - mean) / std
        if label:
            label = np.array(label).astype(np.float32) / 255

            if self.aug:
                image, label = self.transform(image, label)

            label = torch.from_numpy(label)
            label = label.unsqueeze(0)
        else:
            if self.aug:
                image, label = self.transform(image, image)


        # image = self.norm(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)

        return image, label

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    data = Mars('.', mode='earth_mars', labeled=True, train_phase=True)
    # print(data.data_root)
    print(len(data))
    print(data.img_list)
    print(data[0])


