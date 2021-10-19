import argparse
import os
from abc import ABC, abstractmethod
from datetime import datetime

from Attacks.BaseAttack import BaseAttack
from Detectors.DetectorEngine import DeterctorEngine
from Ulitities.Image import Picture


class BaseIterativeAttack(BaseAttack, ABC):
    name = "Base Iterative Attack"

    def __init__(self, detector: DeterctorEngine, steps: int,
                 plot_interval: int = 5, additive_attack=True, root_debug: str = "./Data/Debug/",
                 verbosity: int = 2):
        """
        :param detector: name of the detector to be used to visualize the results
        :param steps: number of attack iterations to perform
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param additive_attack: showl we feed the result of the iteration i as the input of the iteration 1+1?
        :param root_debug: root folder insede which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(detector, root_debug, verbosity)

        assert (steps > 0)

        self.steps = steps

        self.plot_interval = plot_interval

        self.additive_attack = additive_attack

        # counter of the attack iterations that have been applied to the image
        self.step_counter = 0


    def setup(self, target_image: Picture, target_image_mask: Picture, source_image: Picture = None,
              source_image_mask: Picture = None,target_forgery_mask : Picture = None):
        """
        :param source_image: image from which we will compute the target representation
        :param source_image_mask: mask of the imae from which we will compute the target representation
        :param target_forgery_mask: mask highlighting the section of the image that should be identified as forged after the attack
        :return:
        """

        super().setup(target_image, target_image_mask,source_image,source_image_mask,target_forgery_mask)

        # create a folder where to store the data generated at each step
        self.steps_debug_folder = os.path.join(str(self.debug_folder), "steps")
        if self.plot_interval > 0:
            os.makedirs(self.steps_debug_folder)

    def execute(self) -> Picture:
        """
        Start the attack pipeline using the data passed in the initialization
        :return: attacked image
        """
        # execute pre-attack operations
        pristine_image = self.target_image

        # execute post-attack operations
        self._on_before_attack()

        # iterate the attack for the given amount of steps
        attacked_image = pristine_image
        for self.step_counter in range(0, self.steps):

            # print logs
            self.write_to_logs("\n### Step: {} ###".format(self.step_counter), force_print=True)
            step_start_time = datetime.now()
            self.write_to_logs(" start at: {}".format(step_start_time), force_print=False)

            # if the attack is not additive, remove the effect of the previous iteration
            if not self.additive_attack:
                attacked_image = pristine_image

            # execute pre-step operations
            self._on_before_attack_step(attacked_image)

            # execute one step of the attack
            attacked_image = self.attack(attacked_image)

            # execute post-step operations
            self._on_after_attack_step(attacked_image)

            self.write_to_logs(" ended at: {}".format(datetime.now()), force_print=False)
            self.write_to_logs(" duration: {}".format(datetime.now() - step_start_time))

        # execute post-attack operations
        self._on_after_attack(attacked_image)

        return attacked_image

    def _on_before_attack_step(self, image: Picture, *args, **kwargs):
        """
        Instructions to perform before the attack step
        :param image: image before the attack step
        :return:
        """
        pass

    def _on_after_attack_step(self, attacked_image: Picture, *args, **kwargs):
        """
        Instructions to perform after the attack step
        :param image: image before the attack step
        :return:
        """
        if self.plot_interval > 0 and (self.step_counter + 1) % self.plot_interval == 0 and not self.clean_execution:
            self.detector.prediction_pipeline(attacked_image,
                                              path=os.path.join(self.steps_debug_folder, str(self.step_counter + 1))
                                              , original_picture=self.target_image, omask=self.target_image_mask,
                                              note=self.step_note())

    def _on_before_attack(self):
        """
        Write parameters to the log and create a visualization of the initial state
        :return:
        """
        super()._on_before_attack()

        self.write_to_logs("Steps: {}".format(self.steps))
        self.write_to_logs("Plot interval: {}".format(self.plot_interval))
        self.write_to_logs("Additive attack: {}".format(self.additive_attack))

        if not self.clean_execution:
            self.detector.prediction_pipeline(self.target_image,
                                              path=os.path.join(self.steps_debug_folder, str(self.step_counter + 1))
                                              , original_picture=self.target_image, omask=self.target_image_mask,
                                              note=self.step_note())

    def step_note(self):
        """
        :return: The note that will be printed on the step visualization
        """
        return "Step:{}".format(self.step_counter + 1)

    @property
    def progress_proportion(self):
        """
        Return the progress percentage of the iterative attack
        :return:
        """
        return self.step_counter / self.steps

    @staticmethod
    def read_arguments(dataset_root) -> tuple:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        attack_parameters,setup_parameters = BaseAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", '--steps', default=50, type=int, help='Number of attack steps to perform')
        parser.add_argument("-pi", '--plot_interval', default=5, type=int,
                            help='how often (# steps) should the step-visualizations be generated?')
        args = parser.parse_known_args()[0]

        attack_parameters["steps"] = int(args.steps)
        attack_parameters["plot_interval"] = int(args.plot_interval)

        return attack_parameters,setup_parameters
