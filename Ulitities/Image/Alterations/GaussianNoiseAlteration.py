import numpy as np

from Ulitities.Image.Alterations.BaseAlteration import BaseAlteration
from Ulitities.Image.Picture import Picture


class GaussianNoiseAlteration(BaseAlteration):
    name = "Gaussian Noise"

    def __init__(self,mean=0,standard_deviation=1,requires_inputs=False):
        self.mean = mean
        self.standard_deviation = standard_deviation
        super().__init__(requires_inputs=requires_inputs)

    def get_inputs(self):
        self.mean = float(input("Mean:"))
        self.standard_deviation = float(input("Standard deviation"))

    def apply(self, picture):
        noise = np.random.normal(self.mean, self.standard_deviation, size=picture.shape)
        return Picture(picture + noise)
