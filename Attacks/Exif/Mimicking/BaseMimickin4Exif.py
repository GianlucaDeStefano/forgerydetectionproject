from abc import ABC

import numpy as np

from Attacks.BaseWhiteBoxAttack import normalize_gradient
from Attacks.Exif.BaseExifAttack import BaseExifAttack
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Utilities.Image.Picture import Picture


class BaseMimicking4Exif(BaseExifAttack, ABC):

    pass
