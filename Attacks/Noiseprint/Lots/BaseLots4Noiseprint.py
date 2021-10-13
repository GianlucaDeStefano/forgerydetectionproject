from abc import ABC

from Attacks.BaseWhiteBoxAttack import normalize_gradient
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Ulitities.Image.Picture import Picture


class BaseLots4Noiseprint(BaseNoiseprintAttack, ABC):

    name = "Base LOTS 4 Noiseprint Attack"

