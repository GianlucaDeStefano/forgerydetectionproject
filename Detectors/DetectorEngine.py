from abc import abstractmethod

from Ulitities.Image.Picture import Picture


class DeterctorEngine():

    def __init__(self, name):
        self._engine = None
        self.name = name

    @abstractmethod
    def detect(self, image: Picture, target_mask: Picture) -> tuple:
        """
        Function returning a tuple containing the heatmap and its segmented equivalent computed
        using the given engine
        :param image: image to analyze
        :param target_mask: the mask that should ideally be produced
        :return: (heatmap,mask)
        """
        raise NotImplementedError

