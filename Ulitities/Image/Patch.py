import numpy as np


class Patch(np.ndarray):

    def __new__(cls, array: np.array, x_indexes: tuple = None, y_indexes: tuple = None, paddings: tuple = (0, 0, 0, 0),
                zero_paddings: tuple = (0, 0, 0, 0)):
        """
        Class to manage patches extracted from a picture
        :param array: array containing the values of the pixels of the picture
        :param x_indexes: 2-d tuple returning the position of the left anr right vertices on the x-axis
        :param y_indexes: 2-d tuple returning the position of the top and bottom vertices on the y-axis
        :param paddings: 4-d tuple containing the "paddings" of the patch along the top, right,bottom,left directions
        """

        if x_indexes is None:
            x_indexes = (0, array.shape[0])

        if y_indexes is None:
            y_indexes = (0, array.shape[1])

        assert (len(x_indexes) == 2)
        assert (len(y_indexes) == 2)
        assert (len(paddings) == 4)

        obj = np.asarray(array).view(cls)

        obj.x_indexes = x_indexes
        obj.y_indexes = y_indexes
        obj.paddings = paddings
        obj.zero_paddings = zero_paddings

        return obj

    @property
    def clean_shape(self):
        """
        Return the shape of the patch without any padding applied
        :return: tuple
        """
        shape = self.shape

        x = self.shape[0] - self.paddings[1] - self.paddings[3]
        y = self.shape[1] - self.paddings[2] - self.paddings[0]

        return (x, y) + shape[2:]

    def get_verticies(self, with_padding=False):
        """
        Return 2 tuples containing the coordinates of the 2 verticies formingnthe square that contains the patch
        in the original image
        :return: (stat x, start y), (end x, end y)
        """

        top_padding, right_padding, bottom_padding, left_padding = self.paddings
        top_0_padding, right_0_padding, bottom_0_padding, left_0_padding = self.zero_paddings

        x_index, x_index_f = self.x_indexes
        y_index, y_index_f = self.y_indexes

        if with_padding:
            x_index = x_index + left_padding - left_0_padding
            x_index_f = x_index_f - right_padding + right_0_padding

            y_index = y_index + top_padding - top_0_padding
            y_index_f = y_index_f - bottom_padding + bottom_0_padding

        else:
            x_index = x_index + left_padding
            x_index_f = x_index_f - right_padding

            y_index = y_index + top_padding
            y_index_f = y_index_f - bottom_padding

        return (x_index, y_index), (x_index_f, y_index_f)

    def no_paddings(self, array=None):
        """
        Return the values of the patch that are not part of the paddings
        :param array:
        :return:
        """
        if not hasattr(array, 'shape'):
            # No array passed,
            array = self

        values = array[self.paddings[3]:self.shape[0] - self.paddings[1],
                 self.paddings[0]:self.shape[1] - self.paddings[2]]

        (x_index, y_index), (x_index_f, y_index_f) = self.get_verticies(False)

        return Patch(values, (x_index, x_index_f), (y_index, y_index_f))

    def add_to_image(self, image, array: np.array = None):
        """
        Given an image add a patch core data (no paddings) to the image in this patch's original coordinates.
        :param image:
        :param array:
        :return:
        """

        # if no array is given
        if array is None:
            # use this patch
            array = self

        array = self.no_paddings(array)

        # get the boundaries of the core data of the patch
        (x_index, y_index), (x_index_f, y_index_f) = self.get_verticies(True)

        # write the patch on the image
        image[x_index:x_index_f, y_index:y_index_f] += array

        return image
