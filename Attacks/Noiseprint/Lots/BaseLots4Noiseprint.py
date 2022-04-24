from abc import ABC

from Attacks.BaseWhiteBoxAttack import normalize_gradient
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Utilities.Image.Picture import Picture


class BaseLots4Noiseprint(BaseNoiseprintAttack, ABC):

    name = "Base LOTS 4 Noiseprint Attack"

    def setup(self, target_image_path: Picture, target_image_mask: Picture, source_image_path: Picture = None,
              source_image_mask: Picture = None, target_forgery_mask: Picture = None):
        super().setup(target_image_path, target_image_mask, target_image_path, target_image_mask, target_forgery_mask)
