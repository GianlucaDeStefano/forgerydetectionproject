from abc import ABC

import numpy as np

from Attacks.BaseWhiteBoxAttack import normalize_gradient
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Utilities.Image.Picture import Picture


class BaseMimicking4Noiseprint(BaseNoiseprintAttack, ABC):

    name = "BaseMimicking4Noiseprint"