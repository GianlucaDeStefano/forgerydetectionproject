import argparse

import numpy as np

from Attacks.BaseAttack import BaseAttack
from Attacks.BaseIterativeAttack import BaseIterativeAttack
from Ulitities.Image.Picture import Picture


class GaussianNoiseAdditionAttack(BaseIterativeAttack):
    """
    This class is used to implement a black box attack that adds gaussian noise to the image
    """

    name = "Gaussian Noise Addition Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, detector:str, steps: int, initial_mean,
                 initial_standard_deviation,
                 final_mean, final_standard_deviation, debug_root: str = "./Data/Debug/",
                 verbosity: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param steps: number of attack iterations to perform
        :param initial_mean: initial value for the mean of the Gaussian distribution from which we will sample the noise
        :param initial_standard_deviation: initial value for the standard deviation of the Gaussian distribution from which we will sample the noise
        :param final_mean: final value for the mean of the Gaussian distribution from which we will sample the noise
        :param final_standard_deviation: initial value for the standard deviation of the Gaussian distribution from which we will sample the noise
        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: verbosity of the logs printed in the console
        """

        super().__init__(target_image, target_image_mask, detector, steps, False, debug_root, verbosity)

        self.initial_mean = initial_mean
        self.initial_standard_deviation = initial_standard_deviation

        self.final_mean = final_mean
        self.final_standard_deviation = final_standard_deviation

    def _on_before_attack(self):
        """
        Write the input parameters into the logs
        :param pristine_image: starting image without any attack on it
        :return: None
        """
        super(BaseIterativeAttack, self)._on_before_attack()

        # write the parameters into the logs
        self.write_to_logs("Initial Mean:{}".format(str(self.initial_mean)))
        self.write_to_logs("initial standard deviation:{}".format(str(self.initial_standard_deviation)))

        self.write_to_logs("Final Mean:{}".format(str(self.final_mean)))
        self.write_to_logs("Final standard deviation:{}".format(str(self.final_standard_deviation)))

    def attack(self, image_to_attack: Picture, *args, **kwargs):
        """
        Add noise to the image and return the result
        :param image_to_attack: image on which to perform the attack
        :param args: none
        :param kwargs: none
        :return: image + gaussian noise
        """
        mean = self.initial_mean + (self.final_mean - self.initial_mean)*self.progress_proportion
        standard_deviation = self.initial_standard_deviation + (self.final_standard_deviation-self.initial_standard_deviation)*self.progress_proportion

        self.write_to_logs("Mean:{:.2f}, Standard deviation:{:.2f}".format(mean,standard_deviation))

        noise = np.random.normal(mean, standard_deviation, size=image_to_attack.shape)
        return Picture((image_to_attack + noise).clip(0,255))

    @staticmethod
    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseIterativeAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument('--initial_mean', default=None, type=float,
                            help='initial value for the mean of the Gaussian distribution from which we will sample '
                                 'the noise')
        parser.add_argument('--initial_standard_deviation', default=None, type=float,
                            help='initial value for the standard deviation of the Gaussian distribution from which we '
                                 'will sample the noise')
        parser.add_argument('--final_mean', default=None, type=float,
                            help='final value for the mean of the Gaussian distribution from which we will sample the '
                                 'noise')
        parser.add_argument('--final_standard_deviation', default=None, type=float,
                            help='final value for the standard deviation of the Gaussian distribution from which we '
                                 'will sample the noise')

        parser.add_argument('--detector', default=None, type=str,
                            help='Select the detector to use to analyze the result')

        args = parser.parse_known_args()[0]

        list = ["initial_mean", "initial_standard_deviation", "final_mean", "final_standard_deviation"]

        for element in list:
            if getattr(args,element) is None:
                kwarg[element] = float(input("Add a (float) value for the variable {}:".format(element)))
            else:
                kwarg[element] = getattr(args,element)

        if args.detector is None:
            kwarg["detector"] = str(input("Which detector should be used? [Noiseprint,Exif]:"))
        else:
            kwarg["detector"] = args.detector

        return kwarg
