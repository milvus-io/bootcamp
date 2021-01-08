import torch.utils.data as data
import numpy as np
import os
import pickle
import h5py
from PIL import Image
from scipy.misc import imread, imresize
from utils.directory import check_exists


class CuhkPedes(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''
    pklname_list = ['train.pkl', 'val.pkl', 'test.pkl']
    h5name_list = ['train.h5', 'val.h5', 'test.h5']

    def __init__(self, image_root, anno_root, split, max_length, transform=None, target_transform=None, cap_transform=None):
        
        self.image_root = image_root
        self.anno_root = anno_root
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform
        self.split = split.lower()

        if not check_exists(self.image_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        if self.split == 'train':
            self.pklname = self.pklname_list[0]
            #self.h5name = self.h5name_list[0]

            with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.train_labels = data['labels']
                self.train_captions = data['caption_id']
                self.train_images = data['images_path']
            #data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            #self.train_images = data_h5py['images']


        elif self.split == 'val':
            self.pklname = self.pklname_list[1]
            #self.h5name = self.h5name_list[1]
            with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.val_labels = data['labels']
                self.val_captions = data['caption_id']
                self.val_images = data['images_path']
            #data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            #self.val_images = data_h5py['images']

        elif self.split == 'test':
            self.pklname = self.pklname_list[2]
            #self.h5name = self.h5name_list[2]

            with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.test_labels = data['labels']
                self.test_captions = data['caption_id']
                self.test_images = data['images_path']

            #data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            #self.test_images = data_h5py['images']

        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        if self.split == 'train':
            img_path, caption, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]
        img_path = os.path.join(self.image_root, img_path)
        img = imread(img_path)
        img = imresize(img, (224,224))
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.cap_transform is not None:
            caption = self.cap_transform(caption)
        caption = caption[1:-1]
        caption = np.array(caption)
        caption, mask = self.fix_length(caption)
        return img, caption, label, mask, img_path

    def fix_length(self, caption):
        caption_len = caption.shape[0]
        if caption_len < self.max_length:
            pad = np.zeros((self.max_length - caption_len, 1), dtype=np.int64)
            caption = np.append(caption, pad)
        return caption, caption_len

    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
