import glob
import os

import cv2
import numpy as np


class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.data_info = self.get_data_paths()

    def get_data_paths(self, extension="png"):
        img_lists = glob.glob(self.data_dir + "/*." + extension)
        img_lists.sort()
        return img_lists

    def get_batches_all(self, batch_size=10, image_size=(200, 200), allchannel=False, shape=(0, -1)):
        images = []
        for counter, filename in enumerate(self.data_info[shape[0]: shape[1]]):
            img = cv2.imread(filename)
            img = self.resize_and_pad(img, image_size)
            if not allchannel:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            images.append(img)

            if (counter + 1) % batch_size == 0:
                yield np.array(images, dtype=np.uint8)
                images.clear()

    def get_batches_train(self, batch_size=10, image_size=(200, 200), allchannel=False):
        shape = (0, int(len(self.data_info) * 0.8))
        return self.get_batches_all(batch_size=batch_size, image_size=image_size, allchannel=allchannel, shape=shape)

    def get_batches_test(self, batch_size=10, image_size=(200, 200), allchannel=False):
        shape = (int(len(self.data_info) * 0.2), -1)
        return self.get_batches_all(batch_size=batch_size, image_size=image_size, allchannel=allchannel, shape=shape)

    def resize_and_pad(self, img, size):
        h, w = img.shape[:2]

        size = size[0:2]
        sh, sw = size
        # interpolation method
        if h & sh or w & sw:  # shrinking image
            interp = cv2.INTER_AREA
        else:  # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w // h

        # padding
        if aspect:  # horizontal image
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape, dtype=np.uint8)
            h_offset = int((w - h) / 2)
            new_img[h_offset:h_offset + h, :, :] = img.copy()

        elif not aspect:  # vertical image
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape, dtype=np.uint8)
            w_offset = int((h - w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img
