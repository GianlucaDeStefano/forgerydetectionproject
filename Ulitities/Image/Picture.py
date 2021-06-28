from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image


class IncompatibeShapeException(Exception):
    def __init__(self, operation, shape):
        super().__init__("Impossible to perform operation {} on image with shape {}".format(operation, shape))


class Picture(np.ndarray):
    """
    Class used to standardize operations on images
    """

    '''Subclass of ndarray MUST be initialized with a numpy array as first argument.
    '''

    def __new__(cls, array: np.ndarray = None, path: str = "", *args, **kwargs):

        if array is None:
            # no preloaded image is given, let's load it
            print(path)
            input_array = Image.open(path).convert('RGB')
            obj = (np.asarray(input_array, np.int)).view(cls)
        else:
            # a preloaded image is given
            obj = array.view(cls)
        obj.path = path
        return obj

    @property
    def one_channel(self, red_weight=0.299, green_weight=0.587, blue_weight=0.114):
        """
        Convert the image to a one channel image, only 3 and one channel images are admitted
        :return: 1 channel version of the image
        """
        if len(self.shape) == 2 or (len(self.shape) == 3 and self.shape[2] == 1):
            return self
        else:
            try:
                return Picture(red_weight * self[:, :, 0] + green_weight * self[:, :, 1] + blue_weight * self[:, :, 2])
            except:
                raise IncompatibeShapeException("'3 to 1 channels'", self.shape)

    @property
    def three_channel(self, red_weight=0.299, green_weight=0.587,blue_weight =0.114):
        """
        Given a mono-channel image, split it into 3 channels according
        :return:
        """

        if len(self.shape) == 3 and self.shape[2] == 3:
            return self
        else:
            ar = np.zeros((self.shape[0], self.shape[1], 3))
            try:
                ar[:, :, 0] = self[:, :] / (3 * red_weight)
                ar[:, :, 1] = self[:, :] / (3 * green_weight)
                ar[:, :, 2] = self[:, :] / (3 * blue_weight)
                return Picture(ar)
            except:
                raise IncompatibeShapeException("'1 to 3 channels'", self.shape)

    def divide_in_patches(self, patch_shape: tuple, padding=(0, 0, 0, 0), force_shape=False):
        """
        Function to divide an image into patches
        :param patch_shape: target shape of each patch
        :param force_shape: strictly produce only patches of the given shape + padding
        :param padding: 4-d tuple indicating if and by how much we should pad diemnsion of the patch where possible,
            the order is the following: top,right,bottom,left
        :return: list of patches
        """

        for element in padding:
            assert (element >= 0)

        assert (len(patch_shape) == 2)

        patches = []

        for x in range(0, self.shape[0], patch_shape[0]):

            for y in range(0, self.shape[1], patch_shape[1]):

                top_padding, right_padding, bottom_padding, left_padding = padding

                x_index = x - left_padding
                y_index = y - top_padding

                x_index_f = x + patch_shape[0] + right_padding
                y_index_f = y + patch_shape[1] + bottom_padding

                if x_index < 0:
                    if force_shape:
                        continue
                    else:
                        left_padding = left_padding + x_index
                        x_index = 0

                if y_index < 0:
                    if force_shape:
                        continue
                    else:
                        top_padding = top_padding + y_index
                        y_index = 0

                if x_index_f > self.shape[0]:
                    if force_shape:
                        continue
                    else:
                        right_padding = max(right_padding - (x_index_f - self.shape[0]), 0)
                        x_index_f = self.shape[0]

                if y_index_f > self.shape[1]:
                    if force_shape:
                        continue
                    else:
                        bottom_padding = max(bottom_padding - (y_index_f - self.shape[1]), 0)
                        y_index_f = self.shape[1]

                if len(self.shape) == 2 or self.shape[2] == 1:
                    values = self[x_index: x_index_f, y_index:y_index_f]
                else:
                    values = self[x_index: x_index_f, y_index:y_index_f, :]

                patch = Patch(values, (x_index, x_index_f), (y_index, y_index_f),
                              (top_padding, right_padding, bottom_padding, left_padding))

                if force_shape:
                    this_shape = (patch_shape[0] + padding[1] + padding[3], patch_shape[1] + padding[0] + padding[2])
                    assert (patch.shape[0] == this_shape[0] and patch.shape[1] == this_shape[1])

                patches.append(patch)

        return patches

    def get_authentic_patches(self, mask, patch_shape: tuple, padding=(0, 0, 0, 0), force_shape=False):

        assert (self.shape[0] == mask.shape[0])
        assert (self.shape[1] == mask.shape[1])

        # divide image and mask into patches
        img_patches = self.divide_in_patches(patch_shape, padding, force_shape)
        mask_patches = mask.divide_in_patches(patch_shape, padding, force_shape)
        assert (len(img_patches) == len(mask_patches))

        # disgard patches containing masked data
        authentic_patches = []
        for i in range(len(img_patches)):
            if mask_patches[i].sum() == 0:
                authentic_patches.append(img_patches[i])

        return authentic_patches

    def save(self, path):

        if self.max() <=1 and self.min() >=0:
            image_array = np.array(self*256, np.uint8)
        else:
            image_array = np.array(self, np.uint8)
        im = Image.fromarray(image_array)
        im.save(path)

    def to_float(self):
        return Picture((self / 256).clip(0,1))

    def to_int(self):
        return Picture(np.rint(self)*255)

    @property
    def image_path(self):
        return self._image_path


from Ulitities.Image.Patch import Patch
