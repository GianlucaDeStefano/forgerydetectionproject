import argparse
import os
import numpy as np
from PIL import Image
from Attacks.BaseIterativeAttack import BaseIterativeAttack
from Utilities.Image.Picture import Picture
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


class JpegCompressionAttack(BaseIterativeAttack):
    """
        This class is used to implement a black box attack that compress the image using the Jpeg algorithm
    """

    name = "Jpeg Compression Attack"

    def __init__(self, target_image: Picture, target_image_mask: Picture, detector:str, steps: int, initial_quality_level, final_quality_level,
                 debug_root: str = "./Data/Debug/", verbosity: bool = True):
        """
        :param target_image: original image on which we should perform the attack
        :param target_image_mask: original mask of the image on which we should perform the attack
        :param steps: number of attack iterations to perform

        :param debug_root: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: verbosity of the logs printed in the console
        """

        super().__init__(target_image, target_image_mask, detector, steps, False, debug_root, verbosity)
        self.initial_quality_level = initial_quality_level
        self.final_quality_level = final_quality_level


    def _on_before_attack(self):
        """
        Write the input parameters into the logs
        :param pristine_image: starting image without any attack on it
        :return: None
        """

        super(BaseIterativeAttack, self)._on_before_attack()

        # write the parameters into the logs
        self.logger_module.info("Initial Quality Level:{}".format(str(self.initial_quality_level)))
        self.logger_module.info("Final quality Level:{}".format(str(self.final_quality_level)))


    def attack(self, image_to_attack: Picture, *args, **kwargs):
        qf = self.initial_quality_level  - int((self.initial_quality_level - self.final_quality_level) * self.progress_proportion)
        self.logger_module.info("quality level: {}".format(qf))

        path = os.path.join(self.debug_folder, 'compressed_image.jpg')
        img = Image.fromarray(np.array(self.target_image, np.uint8))
        img.save(path, quality=qf)

        if isinstance(self.detector,NoiseprintVisualizer):
            self.detector.load_quality(max(51,min(101,qf)))

        return Picture(path=path)


    @staticmethod
    def read_arguments(dataset_root) -> tuple:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        attack_parameters,setup_parameters = BaseIterativeAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument('--initial_quality_level', default=100, type=int,
                            help='initial (highest) quality level')
        parser.add_argument('--final_quality_level', default=None, type=int,
                            help='final (lowest) quality level')
        parser.add_argument('--detector', default=None, type=str,
                            help='Select the detector to use to analyze the result')

        args = parser.parse_known_args()[0]

        list = ["initial_quality_level", "final_quality_level"]

        for element in list:
            if getattr(args,element) is None:
                attack_parameters[element] = float(input("Add an (int) value [100,1] for the variable {}:".format(element)))
            else:
                attack_parameters[element] = getattr(args,element)

        if args.detector is None:
            attack_parameters["detector"] = str(input("Which detector should be used? [Noiseprint,Exif]:"))
        else:
            attack_parameters["detector"] = args.detector

        return attack_parameters,setup_parameters
