# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from six.moves import cPickle

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


fpath = '/Users/mvccouto/Documents/Mestrado/dataset/'
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

    num_train_samples = len(malignant) + len(benign) + len(normal)

    x_train = np.empty((num_train_samples, 2, 299, 299), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    k = 0
    for i in range(0, len(normal)):
        fname = normal[i]
        fname1 = fpath+classes[2]+mlo+fname
        fname2 = fpath+classes[2]+cc+fname
        x_train[k+i, 0, :, :], x_train[k+i, 1, :, :] = load_images(fname1, fname2)
        y_train[i] = 0

    k += i
    for i in range(0, len(benign)):
        fname = benign[i]
        fname1 = fpath+classes[1]+mlo+fname
        fname2 = fpath+classes[1]+cc+fname
        x_train[k+i, 0, :, :], x_train[k+i, 1, :, :] = load_images(fname1, fname2)
        y_train[i] = 1

    k += i
    for i in range(0, len(malignant)):
        fname = malignant[i]
        fname1 = fpath+classes[0]+mlo+fname
        fname2 = fpath+classes[0]+cc+fname
        x_train[k+i, 0, :, :], x_train[k+i, 1, :, :] = load_images(fname1, fname2)
        y_train[i] = 2

        # plt.imshow(x_train[i, 1, :, :], cmap='gray')
        # plt.show()

    # fpath = os.path.join(path, 'test_batch')
    # x_test, y_test = load_batch(fpath)
    #
    # y_train = np.reshape(y_train, (len(y_train), 1))
    # y_test = np.reshape(y_test, (len(y_test), 1))
    #
    # if K.image_data_format() == 'channels_last':
    #     x_train = x_train.transpose(0, 2, 3, 1)
    #     x_test = x_test.transpose(0, 2, 3, 1)

    return []
    # return (x_train,  y_train), (x_test, y_test)
