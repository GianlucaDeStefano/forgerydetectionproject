from abc import abstractmethod, ABC

from Ulitities.Image.Picture import Picture


class BaseAlteration(ABC):

    def __init__(self, requires_inputs=False):
        self.requires_inputs = requires_inputs
        if self.requires_inputs:
            print("Parameters for {}:".format(self.name))
            self.get_inputs()

    @abstractmethod
    def get_inputs(self):
        raise NotImplementedError

    @abstractmethod
    def apply(self, image: Picture) -> Picture:
        raise NotImplementedError
