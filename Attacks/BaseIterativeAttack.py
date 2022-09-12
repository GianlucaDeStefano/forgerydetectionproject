import argparse
import os
from abc import ABC, abstractmethod
from datetime import datetime

from Attacks.BaseAttack import BaseAttack
from Utilities.Image import Picture
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer


class BaseIterativeAttack(BaseAttack, ABC):
    name = "Base Iterative Attack"

    def __init__(self, visualizer: BaseVisualizer, steps: int,
                 plot_interval: int = 5, additive_attack=True, debug_root: str = "./Data/Debug/",
                 verbosity: int = 2):
        """
        :param visualizer: instance of the visualizer class wrapping the functionalities of the targeted detector
        :param steps: number of attack iterations to perform
        :param plot_interval: how often (# steps) should the step-visualizations be generated?
        :param additive_attack: showl we feed the result of the iteration i as the input of the iteration 1+1?
        :param debug_root: root folder inside which to create a folder to store the data produced by the pipeline
        :param verbosity: is this a test mode? In test mode visualizations and superfluous steps will be skipped in favour of a
            faster execution to test the code
        """

        super().__init__(visualizer, debug_root, verbosity)

        self.steps_debug_folder = None
        assert (steps > 0)

        self.steps = steps

        self.plot_interval = plot_interval

        self.additive_attack = additive_attack

        # counter of the attack iterations that have been applied to the image
        self.step_counter = 0

    def setup(self, target_image_path: Picture, target_image_mask: Picture):
        """
        Setup the pipeline for execution
        @param target_image_path: path fo the sample to process
        @param target_image_mask: np.array containing a binary mask where 0 -> pristine 1-> forged pixel
        @return:
        """

        super().setup(target_image_path, target_image_mask)

        # create a folder where to store the data generated at each step
        self.steps_debug_folder = os.path.join(str(self.debug_folder), "steps")

        os.makedirs(self.steps_debug_folder)

    def execute(self) -> tuple:
        """
        Start the attack pipeline using the data passed in the initialization
        :return: last attacked image,best attacked image
        """
        # execute pre-attack operations
        pristine_image = self.target_image

        # execute post-attack operations
        self._on_before_attack()

        best_image = None

        best_loss = float('inf')

        # iterate the attack for the given amount of steps
        attacked_image = pristine_image
        for self.step_counter in range(0, self.steps):

            # print logs
            step_start_time = datetime.now()
            self.logger_module.info("\n### Step: {} ###".format(self.step_counter))
            self.logger_module.info(" start at: {}".format(step_start_time))

            # if the attack is not additive, remove the effect of the previous iteration
            if not self.additive_attack:
                attacked_image = pristine_image

            # execute pre-step operations
            self._on_before_attack_step(attacked_image)

            # execute one step of the attack
            attacked_image, loss = self.attack(attacked_image)

            # save result with the lowest loss
            if loss < best_loss:
                best_loss = loss
                best_image = attacked_image.copy()

            # execute post-step operations
            self._on_after_attack_step(attacked_image)

            self.logger_module.info(" ended at: {}".format(datetime.now()))
            self.logger_module.info(" duration: {}".format(datetime.now() - step_start_time))

        # execute post-attack operations
        self._on_after_attack(attacked_image)

        return attacked_image, best_image

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

        self.visualizer.initialize(sample=attacked_image, reset_instance=False, reset_metadata=False)

        if self.plot_interval > 0 and (self.step_counter + 1) % self.plot_interval == 0 and not self.test:
            self.visualizer.save_prediction_pipeline(
                path=os.path.join(self.steps_debug_folder, str(self.step_counter + 1)))

    def _on_before_attack(self):
        """
        Write parameters to the log and create a visualization of the initial state
        :return:
        """
        super()._on_before_attack()

        self.logger_module.info("Steps: {}".format(self.steps))
        self.logger_module.info("Plot interval: {}".format(self.plot_interval))
        self.logger_module.info("Additive attack: {}".format(self.additive_attack))

        if not self.test:
            self.visualizer.save_prediction_pipeline(
                path=os.path.join(self.steps_debug_folder, str(self.step_counter + 1)))

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
        attack_parameters, setup_parameters = BaseAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", '--steps', default=50, type=int, help='Number of attack steps to perform')
        parser.add_argument("-pi", '--plot_interval', default=5, type=int,
                            help='how often (# steps) should the step-visualizations be generated?')
        args = parser.parse_known_args()[0]

        attack_parameters["steps"] = int(args.steps)
        attack_parameters["plot_interval"] = int(args.plot_interval)

        return attack_parameters, setup_parameters
