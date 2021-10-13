import argparse

from pathlib import Path
import numpy as np
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Attacks.Noiseprint.Mimiking.BaseMimickin4Noiseprint import BaseMimicking4Noiseprint
from Datasets import get_image_and_mask, ImageNotFoundError
from Detectors.Noiseprint.utility.utility import prepare_image_noiseprint
from Ulitities.Image.Picture import Picture


class NoiseprintMimickingAttack(BaseMimicking4Noiseprint):
    name = "Noiseprint mimicking attack"

    def _compute_target_representation(self, target_representation_source_image: Picture,
                                       target_representation_source_image_mask: Picture):
        """
            This type of attack tries to "paste" noiseprint generated fon an authentic image on top of a
            forged one. The target representation is simply the noiseprint of the authentic image
        """
        image = prepare_image_noiseprint(target_representation_source_image)

        # generate an image wise noiseprint representation on the entire image
        original_noiseprint = Picture(self._engine.predict(image))
        return original_noiseprint



    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseNoiseprintAttack.read_arguments(dataset_root)

        parser = argparse.ArgumentParser()
        parser.add_argument('--source_image', required=True,
                            help='Name of the image to use as source for the noiseprint')
        args = parser.parse_known_args()[0]

        image_path = args.source_image

        try:
            image, mask = get_image_and_mask(dataset_root, image_path)
        except ImageNotFoundError:
            # the image is not present in the dataset, look if a direct reference has been given
            image_path = Path(image_path)

            if image_path.exists():
                image = Picture(str(image_path))
                mask = np.where(np.all(image == (255,255,255), axis=-1), 1, 0)
            else:
                raise

        kwarg["source_image"] = image
        kwarg["source_image_mask"] = mask
        return kwarg
