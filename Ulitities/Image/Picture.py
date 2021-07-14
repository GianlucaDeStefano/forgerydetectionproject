import numpy as np
from PIL import Image

from Ulitities.Image.Patch import Patch


class IncompatibeShapeException(Exception):
    def __init__(self, operation, shape):
        super().__init__("Impossible to perform operation {} on image with shape {}".format(operation, shape))


class Picture(Patch):
    """
    Class used to standardize operations on images
    """

    def __new__(cls, value: np.ndarray = None, path: str = "", *args, **kwargs):

        if value is not None and type(value) is str:
            path = str(value)
            value = None

        # if no numpy array is directly given
        if value is None:

            # if no numpy array or path is given raise an exception
            if not path:
                raise Exception("Impossible to instantiate a Picture class without a numpy array or a path to load "
                                "it from")

            # use the path to load the picture as a
            value = np.asarray(Image.open(path).convert('RGB'), np.int)

        obj = Patch.__new__(cls, value, *args, **kwargs)
        obj._path = path

        return obj

    @property
    def one_channel(self, red_weight=0.299, green_weight=0.587, blue_weight=0.114):
        """
        Convert the image to a one channel image
        :return: 1 channel version of the image
        """
        if len(self.shape) == 2 or (len(self.shape) == 3 and self.shape[2] == 1):
            return self
        else:

            return Picture(red_weight * self[:, :, 0] + green_weight * self[:, :, 1] + blue_weight * self[:, :, 2])

    @property
    def three_channels(self, red_weight=0.299, green_weight=0.587, blue_weight=0.114):
        """
        Given a single channel numpy array, convert into a 3 channel array according to the given weights (the default weights
        are the one used to produce the luminance from an RGB image) :return:
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

    def divide_in_patches(self, patch_shape: tuple, padding=(0, 0, 0, 0), force_shape=False, zero_padding=True):
        """
        Function to divide an image into patches
        :param patch_shape: target shape of each patch
        :param force_shape: strictly produce only patches of the given shape + padding without using 0 padding
        :param padding: 4-d tuple indicating if and by how much we should pad dimension of the patch where possible,
            the order is the following: top,right,bottom,left
        :param zero_padding: if there aren't enoug pixels to complete the patch and/or the margin, fill the missin pixes with
            zeros
        :return: list of patches
        """

        for element in padding:
            assert (element >= 0)

        assert (len(patch_shape) == 2)

        patches = []

        # iterate over the x axis
        for x_step in range(0, self.shape[0], patch_shape[0]):

            # iteraste over the y axis
            for y_step in range(0, self.shape[1], patch_shape[1]):
                target_top_padding, target_right_padding, taret_bottom_padding, target_left_padding = padding

                # variables to save how much 0-padding we add along each dimension
                top_0_padding = 0
                right_0_padding = 0
                bottom_0_padding = 0
                left_0_padding = 0

                # variables to save how much "real" padding there is along each dimension
                top_padding, right_padding, bottom_padding, left_padding = padding

                # compute start and end index of the batch on the x axis
                x_start = x_step - left_padding
                x_end = (x_step + patch_shape[0]) + right_padding

                # compute start and end index of the batch on the x axis
                y_start = y_step - top_padding
                y_end = (y_step + patch_shape[1]) + bottom_padding

                # if the patch overflows along one or more axis and we require a stric shape without 0-paddings go to the next patch
                if force_shape:
                    if x_start < 0 or y_start < 0 or x_end > self.shape[0] or y_end > self.shape[1]:
                        continue
                else:
                    # compute starting indexes and relative paddings
                    if x_start < 0:
                        left_0_padding = - x_start
                        left_padding = x_start + target_left_padding
                        x_start = 0

                    if y_start < 0:
                        top_0_padding = - y_start
                        top_padding = y_start + target_top_padding
                        y_start = 0

                    if x_end > self.shape[0]:
                        right_0_padding = x_end - self.shape[0]
                        right_padding = max(right_padding - (x_end - self.shape[0]), 0)
                        x_end = self.shape[0]

                    if y_end > self.shape[1]:
                        bottom_0_padding = y_end - self.shape[1]
                        bottom_padding = max(bottom_padding - (y_end - self.shape[1]), 0)
                        y_end = self.shape[1]

                # if no 0 padding
                if not zero_padding:
                    # reset 0-padding varibles to correctly compute the shape without them
                    top_0_padding = 0
                    right_0_padding = 0
                    bottom_0_padding = 0
                    left_0_padding = 0

                channels = 1
                if (len(self.shape) > 2):
                    channels = self.shape[2]

                shape = (left_0_padding + x_end - x_start + right_0_padding,
                         top_0_padding + y_end - y_start + bottom_0_padding,
                         channels)

                # force shape and 0 padding have both to produce a patch of shape equals to patch_shape
                if force_shape or zero_padding:
                    # let's make sure this is the case
                    assert (shape[0] == patch_shape[0] + target_right_padding + target_left_padding and shape[1] ==
                            patch_shape[1] + taret_bottom_padding + target_top_padding)

                # create an object to hold the value sof the patch
                values = np.squeeze(np.zeros(shape))
                # write the values in the output buffer

                t1 = -right_0_padding
                if t1 == 0:
                    t1 = values.shape[0]

                t2 = -bottom_0_padding
                if t2 == 0:
                    t2 = values.shape[1]

                values[left_0_padding:t1, top_0_padding:t2] = self[x_start: x_end, y_start:y_end]

                # create patch object
                patch = Picture(values, "", (x_start, x_end), (y_start, y_end),
                                (top_padding + top_0_padding, right_padding + right_0_padding,
                                 bottom_padding + bottom_0_padding, left_padding + left_0_padding),
                                (top_0_padding, right_0_padding, bottom_0_padding, left_0_padding))

                patches.append(patch)

        return patches

    def get_authentic_patches(self, mask, patch_shape: tuple, padding=(0, 0, 0, 0), force_shape=False,
                              zero_padding=True):

        """
        Divide the image into patches, returning only the authentic ones, we define a patch authenic if it contains no
        forged pixels
        :param mask: binary int mask defining 0-> authentic pixels 1-> forged pixels
        :param patch_shape: shape of the patches
        :param padding: padding of each patch (the padding may also contain forged pixels)
        :param force_shape: true if we want only patches strictly of the desied shape
        :param zero_padding: true and were no padding is available 0 padding will be used to fill the missing sections
        :param debug_folder:
        :return:
        """

        assert (self.shape[0] == mask.shape[0])
        assert (self.shape[1] == mask.shape[1])

        # divide image and mask into patches
        img_patches = self.divide_in_patches(patch_shape, padding, force_shape, zero_padding)
        mask_patches = mask.divide_in_patches(patch_shape, padding, force_shape, zero_padding)
        assert (len(img_patches) == len(mask_patches))

        recomposed_mask = np.zeros(self.one_channel.shape)

        # disgard patches containing masked data
        authentic_patches = []
        for i in range(len(img_patches)):
            if mask_patches[i].no_paddings().sum() == 0:
                authentic_patches.append(img_patches[i])
            else:
                mask_patches[i].add_to_image(recomposed_mask)

        return authentic_patches

    def get_forged_patches(self, mask, patch_shape: tuple, padding=(0, 0, 0, 0), force_shape=False, zero_padding=True):
        """
        Divide the image into patches, returning only the forged ones, we define a patch forgered if it contains no
        authentic pixels
        :param mask: binary int mask defining 0-> authentic pixels 1-> forged pixels
        :param patch_shape: shape of the patches
        :param padding: padding of each patch (the padding may also contain forged pixels)
        :param force_shape: true if we want only patches strictly of the desied shape
        :param zero_padding: true and were no padding is available 0 padding will be used to fill the missing sections
        :param debug_folder:
        :return:
        """
        assert (self.shape[0] == mask.shape[0])
        assert (self.shape[1] == mask.shape[1])

        # divide image and mask into patches
        img_patches = self.divide_in_patches(patch_shape, padding, force_shape, zero_padding)
        mask_patches = mask.divide_in_patches(patch_shape, padding, force_shape, zero_padding)
        assert (len(img_patches) == len(mask_patches))

        recomposed_mask = np.zeros(self.one_channel.shape)

        # disgard patches containing true data
        forged_patches = []
        for i in range(len(img_patches)):
            if np.all(mask_patches[i].no_paddings() == 1):
                forged_patches.append(img_patches[i])
            else:
                recomposed_mask = mask_patches[i].add_to_image(recomposed_mask)

        return forged_patches

    def save(self, path):

        if self.max() <= 1 and self.min() >= 0:
            image_array = np.array(self * 255, np.uint8)
        else:
            image_array = np.array(self, np.uint8)
        im = Image.fromarray(image_array)
        im.save(path)

    def to_float(self):
        return Picture((self / 255).clip(0, 1), self.path)

    def to_int(self):
        return Picture(np.rint(self) * 255, self.path)

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return Picture(np.array(self).astype(dtype, order, casting, subok, copy), self.path)

    @property
    def path(self):
        if hasattr(self, "_path") and self._path:
            return self._path
        return ""
