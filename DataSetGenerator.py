import glob
import os

import cv2
import numpy as np
import os


def resize_and_pad(img, size):
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


class _DataSetGenerator:
    def __init__(self, data_dir, extension="jpg", class_name=None):
        self.__data_dir = os.path.abspath(data_dir)
        self.__class_name = class_name
        self.__extension = extension
        self.__data_info = self.get_data_paths()

    def get_data_paths(self):
        img_lists = glob.glob(self.__data_dir + "/*." + self.__extension)
        img_lists.sort()
        return img_lists

    def get_batches_all(self, batch_size=10, image_size=(200, 200), allchannel=False, range=(0, -1)):
        images = []
        classes = []
        image_list = self.__data_info[range[0]: range[1]]
        for counter, filename in enumerate(image_list):
            img = cv2.imread(filename)
            if img is None:
                continue

            img = resize_and_pad(img, image_size)
            if not allchannel:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            images.append(img)
            classes.append(self.__class_name)

            if (counter + 1) % batch_size == 0:
                yield (np.array(classes), np.array(images, dtype=np.uint8))
                images.clear()
                classes.clear()

        if len(images) != 0:
            yield (np.array(classes), np.array(images, dtype=np.uint8))

    def get_batches_train(self, batch_size=10, image_size=(200, 200), allchannel=False):
        range = (0, int(len(self.__data_info) * 0.8))
        return self.get_batches_all(batch_size=batch_size, image_size=image_size, allchannel=allchannel, range=range)

    def get_batches_test(self, batch_size=10, image_size=(200, 200), allchannel=False):
        shape = (int(len(self.__data_info) * 0.8), -1)
        return self.get_batches_all(batch_size=batch_size, image_size=image_size, allchannel=allchannel, range=shape)


class DataSet:
    class_cnt = 0

    def __init__(self, batch_size=10, image_size=(200, 200), allchannel=False):
        self.__classes = list()
        self.__batch_size = batch_size
        self.__image_size = image_size
        self.__allchannel = allchannel

    def add_class(self, path, extension="jpg"):
        self.__classes.append(_DataSetGenerator(path, extension, self.class_cnt))
        self.class_cnt += 1

    def get_batches_train(self,):
        generators = list()
        for class_ in self.__classes:
            generators.append(class_.get_batches_train(self.__batch_size, self.__image_size, self.__allchannel))

        for generator in generators:
            for y, x in generator:
                yield (y, x)

    def get_batches_test(self,):
        generators = list()
        for class_ in self.__classes:
            generators.append(class_.get_batches_test(self.__batch_size, self.__image_size, self.__allchannel))

        for generator in generators:
            for y, x in generator:
                yield (y, x)

    def get_data_paths(self):
        class_path = list()
        for class_ in self.__classes:
            class_path.extend(class_.get_data_paths())

        return class_path

    def get_random_example(self):
        paths = self.get_data_paths()
        concrete_path = paths[np.random.randint(0, len(paths))]
        return resize_and_pad(cv2.imread(concrete_path), self.__image_size)
