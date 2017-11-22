# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from six.moves import cPickle
import keras.backend as K
from keras.utils import Sequence

def load_images(fpath1, fpath2):
    img1 = np.asarray(Image.open(fpath1))
    img2 = np.asarray(Image.open(fpath2))
    return img1, img2


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


fpath = '/home/mvccouto/Documentos/Mestrado/estudo_dirigido_i/ddsm/dataset128/'
cc = 'cc/'
mlo = 'mlo/'
classes = ['malignant/', 'benign/', 'normal/']

def load_data_info():
    data_info = []
    for clz in classes:
        dir1 = fpath + clz + cc
        dir2 = fpath + clz + mlo

        fnames = list(set(os.listdir(dir1)) & set(os.listdir(dir2)))
        data_info.append(fnames)
    return tuple(data_info)


def load_data():
    (malignant, benign, normal) = load_data_info()
    malignant = malignant[0:10]
    benign = benign[0:10]
    normal = normal[0:10]

    num_malignant = len(malignant)
    num_benign = len(benign)
    num_normal = len(normal)

    num_train_malignant = math.floor(num_malignant*0.7)
    num_train_benign = math.floor(num_benign*0.7)
    num_train_normal = math.floor(num_normal*0.7)

    num_train_samples = num_train_benign + num_train_malignant + num_train_normal
    num_test_samples = num_normal + num_malignant + num_benign - num_train_samples

    x_train = np.empty((num_train_samples, 2, 128, 128), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    k = 0
    for i in range(0, num_train_normal):
        fname = normal[i]
        fname1 = fpath+classes[2]+mlo+fname
        fname2 = fpath+classes[2]+cc+fname
        x_train[k, 0, :, :], x_train[k, 1, :, :] = load_images(fname1, fname2)
        y_train[k] = 0
        k += 1

    for i in range(0, num_train_benign):
        fname = benign[i]
        fname1 = fpath+classes[1]+mlo+fname
        fname2 = fpath+classes[1]+cc+fname
        x_train[k, 0, :, :], x_train[k, 1, :, :] = load_images(fname1, fname2)
        y_train[k] = 1
        k += 1

    for i in range(0, num_train_malignant):
        fname = malignant[i]
        fname1 = fpath+classes[0]+mlo+fname
        fname2 = fpath+classes[0]+cc+fname
        x_train[k, 0, :, :], x_train[k, 1, :, :] = load_images(fname1, fname2)
        y_train[k] = 2
        k += 1

    x_test = np.empty((num_test_samples, 2, 128, 128), dtype='uint8')
    y_test = np.empty((num_test_samples,), dtype='uint8')

    k = 0
    for i in range(num_train_normal, num_normal):
        fname = normal[i]
        fname1 = fpath+classes[2]+mlo+fname
        fname2 = fpath+classes[2]+cc+fname
        x_test[k, 0, :, :], x_test[k, 1, :, :] = load_images(fname1, fname2)
        y_test[k] = 0
        k += 1

    for i in range(num_train_benign, num_benign):
        fname = benign[i]
        fname1 = fpath+classes[1]+mlo+fname
        fname2 = fpath+classes[1]+cc+fname
        x_test[k, 0, :, :], x_test[k, 1, :, :] = load_images(fname1, fname2)
        y_test[k] = 1
        k += 1

    for i in range(num_train_malignant, num_malignant):
        fname = malignant[i]
        fname1 = fpath+classes[0]+mlo+fname
        fname2 = fpath+classes[0]+cc+fname
        x_test[k, 0, :, :], x_test[k, 1, :, :] = load_images(fname1, fname2)
        y_test[k] = 2
        k += 1

        # plt.imshow(x_train[i, 1, :, :], cmap='gray')
        # plt.show()

    y_train = np.reshape(y_train, (len(y_train),))
    y_test = np.reshape(y_test, (len(y_test),))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train,  y_train), (x_test, y_test)


class DDSMSequence(Sequence):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        (self.malignant, self.benign, self.normal) = load_data_info()

        self.num_malignant = len(self.malignant)
        self.num_benign = len(self.benign)
        self.num_normal = len(self.normal)

        self.num_train_malignant = math.floor(self.num_malignant*0.7)
        self.num_train_benign = math.floor(self.num_benign*0.7)
        self.num_train_normal = math.floor(self.num_normal*0.7)

        self.num_train_samples = self.num_train_benign + self.num_train_malignant + self.num_train_normal

        self.files = []
        for i in range(0, num_train_normal):
            fname = normal[i]
            fname1 = fpath+classes[2]+mlo+fname
            fname2 = fpath+classes[2]+cc+fname
            self.files.append((fname1, fname2, 0))

        for i in range(0, num_train_benign):
            fname = benign[i]
            fname1 = fpath+classes[1]+mlo+fname
            fname2 = fpath+classes[1]+cc+fname
            self.files.append((fname1, fname2, 1))

        for i in range(0, num_train_malignant):
            fname = malignant[i]
            fname1 = fpath+classes[0]+mlo+fname
            fname2 = fpath+classes[0]+cc+fname
            self.files.append((fname1, fname2, 2))

    def __len__(self):
        return math.ceil(len(self.num_train_samples) / self.batch_size)

    def __getitem__(self, idx):
        x_train = np.empty((self.batch_size, 2, 128, 128), dtype='uint8')
        y_train = np.empty((self.batch_size,), dtype='uint8')

        batch_x = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        k = 0
        for (fname1, fname2, y) in batch_x:
            x_train[k, 0, :, :], x_train[k, 1, :, :] = load_images(fname1, fname2)
            y_train[k] = y
            k += 1

        y_train = np.reshape(y_train, (len(y_train), 1))
        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)

            # [3,3,2,32], [32,3,3,3]

        return x_train, y_train
