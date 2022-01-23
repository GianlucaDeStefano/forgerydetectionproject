from abc import abstractmethod

from Utilities.Image.Picture import Picture
from Utilities.Logger.Logger import Logger


class DeterctorEngine(Logger):
    name = ""

    def __init__(self, name):
        super().__init__()
        self._engine = None
        DeterctorEngine.name = name

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

