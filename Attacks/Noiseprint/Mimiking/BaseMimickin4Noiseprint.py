from abc import ABC

import numpy as np

from Attacks.BaseWhiteBoxAttack import normalize_gradient
from Attacks.Noiseprint.BaseNoiseprintAttack import BaseNoiseprintAttack
from Ulitities.Image.Picture import Picture


class BaseMimicking4Noiseprint(BaseNoiseprintAttack, ABC):

    def attack(self, image_to_attack: Picture, *args, **kwargs):
        """
        Perform step of the attack executing the following steps:
            (2) -> compute the gradient
            (3) -> normalize the gradient
            (4) -> apply the gradient to the image with the desired strength
            (5) -> return the image
        :return: attacked image
        """

        # apply Nesterov momentum
        image_to_attack = np.array(image_to_attack, dtype=float) - self.moving_avg_gradient

        # compute the gradient
        image_gradient, loss = self._get_gradient_of_image(image_to_attack, self.target_representation,
                                                           Picture(self.noise))

        # save loss value to plot it
        self.loss_steps.append(loss)

        # compute the decaying alpha
        alpha = self.alpha / (1 + 0.05 * self.step_counter)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, self.gradient_normalization_margin) * alpha

        # update the moving average
        self.moving_avg_gradient = self.moving_avg_gradient * self.momentum_coeficient + (
                    1 - self.momentum_coeficient) * image_gradient

        # add this iteration contribution to the cumulative noise
        self.noise += self.moving_avg_gradient / (1 - self.momentum_coeficient ** (1 + self.step_counter))

        return self.attacked_image