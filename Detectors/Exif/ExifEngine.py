import os
import pathlib

from Detectors.DetectorEngine import DeterctorEngine
from Detectors.Exif import demo
from Ulitities.Image.Picture import Picture


class ExifEngine(DeterctorEngine):

    def __init__(self):
        super().__init__("ExifEngine")

        ckpt_path = os.path.join(pathlib.Path(__file__).parent,'./ckpt/exif_final/exif_final.ckpt')
        self.model = demo.Demo(ckpt_path=ckpt_path, use_gpu=0, quality=3.0, num_per_dim=30)

    def detect(self, image: Picture):
        res = self.model.run(image, use_ncuts=True, blue_high=True)
        return res[0], res[1]
