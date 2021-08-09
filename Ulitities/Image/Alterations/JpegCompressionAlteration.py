import io
import imageio

from Ulitities.Image.Alterations.BaseAlteration import BaseAlteration
from Ulitities.Image.Picture import Picture


class JpegCompressionAlteration(BaseAlteration):
    name = "Jpeg compression"

    def __init__(self,quality=100,requires_inputs=False):
        self.quality = quality
        super().__init__(requires_inputs=requires_inputs)

    def get_inputs(self):
        self.quality = int(input("Quality:"))
        assert (0 < self.quality < 102)

    def apply(self, picture):
        buf = io.BytesIO()
        imageio.imwrite(buf, picture, format='jpeg', quality=self.quality,subsampling=0)
        s = buf.getbuffer()
        return Picture(imageio.imread(s, format='jpeg'))
