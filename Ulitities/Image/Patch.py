import numpy as np

from Ulitities.Image.Picture import Picture


class Patch(Picture):

    def __new__(cls, array: np.array, x_indexes: tuple, y_indexes: tuple, paddings: tuple = (0, 0, 0, 0)):
        """
        Class to manage patches extracted from a picture
        :param array: array containing the values of the pixels of the picture
        :param x_indexes: 2-d tuple returning the position of the left anr right vertices on the x-axis
        :param y_indexes: 2-d tuple returning the position of the top and bottom vertices on the y-axis
        :param paddings: 4-d tuple containing the "paddings" of the patch along the top, right,bottom,left directions
        """

        if not paddings:
            paddings = (0, 0, 0, 0)

        assert (len(x_indexes) == 2)
        assert (len(y_indexes) == 2)
        assert (len(paddings) == 4)

        obj = np.asarray(array).view(cls)

        obj.x_indexes = x_indexes
        obj.y_indexes = y_indexes
        obj.paddings = paddings

        return obj

    @property
    def clean_shape(self):
        """
        Return the shape of the patch without any padding applied
        :return: tuple
        """

        if len(self.shape) == 2:
            return (self.shape[0] - self.paddings[1] - self.paddings[3],
                    self.shape[1] - self.paddings[2] - self.paddings[0])
        else:
            return (self.shape[0] - self.paddings[1] - self.paddings[3],
                    self.shape[1] - self.paddings[2] - self.paddings[0], self.shape[2])

    def no_paddings(self,array = None):

        if not hasattr(array, 'shape'):
            # No array passed,
            array = self

        if len(array.shape) == 2:
            return array[self.paddings[3]:self.shape[0] - self.paddings[1],
                   self.paddings[0]:self.shape[1] - self.paddings[2]]
        else:
            return array[self.paddings[3]:self.shape[0] - self.paddings[1],
                   self.paddings[0]:self.shape[1] - self.paddings[2], :]

    def enlarge(self, shape):
        """
        Enlarge patch to the desired size filling gaps with zeros
        :param shape: shape of the final patch
        :return: patch of the desired shape
        """
        top_padding, right_padding, bottom_padding, left_padding = self.paddings

        x_index, x_index_f = self.x_indexes
        y_index, y_index_f = self.y_indexes

        x_index = x_index + left_padding
        x_index_f = x_index_f - right_padding

        y_index = y_index + top_padding
        y_index_f = y_index_f - bottom_padding

        picture = np.zeros(shape)

        picture[x_index:x_index_f, y_index:y_index_f] = self.no_paddings()
        return picture

    def add_to_image(self, image, array: np.array = None):
        top_padding, right_padding, bottom_padding, left_padding = self.paddings

        if not hasattr(array, 'shape'):
            # No array passed,
            array = self

        x_index, x_index_f = self.x_indexes
        y_index, y_index_f = self.y_indexes

        x_index = x_index + left_padding
        x_index_f = x_index_f - right_padding

        y_index = y_index + top_padding
        y_index_f = y_index_f - bottom_padding

        image[x_index:x_index_f, y_index:y_index_f] = self.no_paddings(array)
        return image

