from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random

class LabelDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category2code.txt')
        self.cat = {}
        self.cate = class_choice

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            if train:
                dir_point = os.path.join(self.root, self.cat[item], 'train', 'points')
                dir_seg = os.path.join(self.root, self.cat[item], 'train', 'points_label')
            else:
                dir_point = os.path.join(self.root, self.cat[item], 'test', 'points')
                dir_seg = os.path.join(self.root, self.cat[item], 'test', 'points_label')
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))
            fns = sorted(fns, key=lambda x: int(x[:-4]))

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.txt'), os.path.join(dir_seg, token + '.txt')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        #self.num_seg_classes = 24
        self.num_seg_classes = 3
        #print(self.num_seg_classes)


    def __getitem__(self, index):
        fn = self.datapath[index]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)
        if self.npoints == len(seg):
            choice = np.array(range(0, len(seg)))
        else:
            incompleted = self.npoints - len(seg)
            choice_incompleted = np.random.choice(len(seg), incompleted, replace=True)
            choice = np.array(range(0, len(seg)))
            choice = np.append(choice, choice_incompleted)
        choice = sorted(choice)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        #return fn[1], point_set, seg
        return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'for2500', class_choice = ['Pistol'])
    print(len(d))
    ps, seg = d[0]
    print(ps,seg)


