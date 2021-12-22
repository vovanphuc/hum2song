import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import random

def graw_head_tail(data, max_head=63, max_tail=63):
    return data[random.randint(0, max_head): data.shape[0] - random.randint(0, max_tail), ...]

def cut_off(data, max_cut=63*4):
    max_cut = max(1, min(data.shape[0] - 63*3, max_cut))
    return data[: data.shape[0] - random.randint(0, max_cut), ...]

def augment(data):
    if random.random() < 0.2:
        data = cut_off(data)
    if random.random() < 0.4:
        data = graw_head_tail(data)
    return data

class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(630, 80), mp3aug_ratio=0.2, npy_aug=True):
        self.phase = phase
        self.input_shape = input_shape
        self.aug_ratio = mp3aug_ratio
        self.npy_aug = npy_aug

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        imgs = np.random.permutation(imgs)

        self.augs = []
        self.orgs = []

        for file in imgs:
            if "aug" in os.path.basename(file):
                self.augs.append(file)
            else:
                self.orgs.append(file)


    def __getitem__(self, index):
        if index >= len(self.orgs):
            sample = random.choice(self.augs)
        else:
            sample = self.orgs[index]
        
        splits = sample.split()
        npy_path = splits[0]

        data = np.load(npy_path)

        data = data[:self.input_shape[0], ...]
        
        if self.npy_aug:
            data = augment(data)

        if data.shape[0] >= self.input_shape[0]:
            result = data[:self.input_shape[0], :]
        else:
            result = np.zeros((self.input_shape[0], self.input_shape[1]))
            result[:data.shape[0], :data.shape[1]] = data
        label = np.int32(splits[1])

        data = torch.from_numpy(result).unsqueeze(0)
        return data.float(), label

    def __len__(self):
        return len(self.orgs) + int(self.aug_ratio * len(self.augs))
