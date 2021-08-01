from abc import abstractmethod

from Ulitities.Image.Picture import Picture


class DeterctorEngine():

    def __init__(self,name):

        self.name = name

    @abstractmethod
    def detect(self,image:Picture):
        """
        Function returning a tuple containing the heatmap and its segmented equivalent computed
        using the given engine
        :param image:
        :return:
        """
        raise  NotImplementedError