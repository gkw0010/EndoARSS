from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
from PIL import Image


class RandomScaleCrop(object):
    """
    Credit to Jialong Wu from https://github.com/lorenmt/mtan/issues/34.
    """
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth, normal):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale) - 1)]
        h, w = int(height / sc), int(width / sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        label_ = F.interpolate(label[None, None, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None, :, i:i + h, j:j + w], size=(height, width), mode='nearest').squeeze(0)
        normal_ = F.interpolate(normal[None, :, i:i + h, j:j + w], size=(height, width), mode='bilinear', align_corners=True).squeeze(0)
        return img_, label_, depth_ / sc, normal_


def get_path(root):
    out = []
    for file in os.listdir(root):
        img_path = root + '/' + file + '/image'
        mask_path = root + '/' + file + '/mask'
        for img_file in os.listdir(mask_path):
            if img_file.split('.')[-1] != 'png':
                continue
            out.append([img_path + '/' + img_file, mask_path + '/' + img_file])
    return out


class DS(Dataset):
    """
    We could further improve the performance with the data augmentation of NYUv2 defined in:
        [1] PAD-Net: Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing
        [2] Pattern affinitive propagation across depth, surface normal and semantic segmentation
        [3] Mti-net: Multiscale task interaction networks for multi-task learning

        1. Random scale in a selected raio 1.0, 1.2, and 1.5.
        2. Random horizontal flip.

    Please note that: all baselines and MTAN did NOT apply data augmentation in the original paper.
    """
    def __init__(self, root, mode='train', augmentation=False, img_size=512, return_name=False):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        if self.mode == 'train':
            self.data_path = self.root + '/train'
        elif self.mode == 'val':
            self.data_path = self.root + '/val'
        else:
            self.data_path = self.root + '/test'

        self.files = get_path(self.data_path)
        self.img_size = img_size
        self.return_name = return_name

    def __getitem__(self, i):
        files = self.files[i]
        img_file, mask_file = files
        #class_label = int(img_file.replace('\\', '/').split('/')[-3]) - 1
        #class_label = np.array(class_label)
        class_label = int(img_file.replace('\\', '/').split('/')[-3]) - 1  # 12个分类中的一个
        class_label = np.array(class_label)
        img = np.array(Image.open(img_file).convert('RGB').resize((self.img_size, self.img_size))) / 255.

        mask = np.array(Image.open(mask_file).convert('L').resize((self.img_size, self.img_size), Image.NEAREST))
        #m = np.unique(mask)[1] - 1
        #
        #mask[mask > 5] -= 2
        #print(mask)
        mask[mask > 0] -= 1
        try:
            assert np.max(mask) <= 9

            img = torch.from_numpy(img.transpose(2, 0, 1)).type(torch.FloatTensor)
            # process
            # mask[mask > 0] += (class_label * 3)
            mask = torch.from_numpy(mask).type(torch.FloatTensor).long()
            class_label = torch.from_numpy(class_label).type(torch.FloatTensor).long()

            if self.return_name:
                return img, {'segmentation': mask, 'classification': class_label, 'path': files}
            return img, {'segmentation': mask, 'classification': class_label}
    
        except Exception as e:
            print(np.max(mask))


    def __len__(self):
        return len(self.files)
