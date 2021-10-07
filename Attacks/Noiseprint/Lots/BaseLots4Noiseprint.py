from abc import ABC

from Attacks.BaseWhiteBoxAttack import normalize_gradient
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Ulitities.Image.Picture import Picture


class BaseLots4Noiseprint(BaseNoiseprintAttack, ABC):

    def attack(self, image_to_attack: Picture, *args, **kwargs):
        """
        Perform step of the attack executing the following steps:
            (1) -> prepare the image to be used by noiseprint
            (2) -> compute the gradient
            (3) -> normalize the gradient
            (4) -> apply the gradient to the image with the desired strength
            (5) -> return the image
        :return: attacked image
        """

        # compute the attacked image using the original image and the cumulative noise to reduce
        # rounding artifacts caused by translating the noise from one to 3 channels and vice versa multiple times
        image_one_channel = Picture((image_to_attack.one_channel() - self.noise).clip(0, 255)).to_float()

        # compute the gradient
        image_gradient, loss = self._get_gradient_of_image(image_one_channel, self.target_representation,
                                                           Picture(self.noise))

        # save loss value to plot it
        self.loss_steps.append(loss)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, self.gradient_normalization_margin) * self.alpha

        # add this iteration contribution to the cumulative noise
        self.noise += image_gradient

        return self.attacked_image
