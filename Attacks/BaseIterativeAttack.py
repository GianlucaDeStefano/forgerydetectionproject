import argparse
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Type

import numpy as np

from Attacks.BaseAttack import BaseAttack
from Datasets import get_image_and_mask
from Ulitities.Image.Picture import Picture
from Ulitities.Visualizers.BaseVisualizer import BaseVisualizer
from Ulitities.Visualizers.ExifVisualizer import ExifVisualizer
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Ulitities.io.folders import create_debug_folder


class BaseIterativeAttack(BaseAttack, ABC):
    name = "BaseIterativveAttack"

    def __init__(self, objective_image: Picture, objective_mask: Picture, steps=50, debug_root="./Data/Debug/",
                 plot_interval=3, verbose=True, step_visualizer: BaseVisualizer = None):
        """
        Base class to implement various attacks
        :param objective_image: image to attack
        :param objective_mask: binary mask of the image to attack, 0 = authentic, 1 = forged
        :param name: name to identify the attack
        :param steps: total number of steps of the attack
        :param debug_root: root folder in which save debug data generated by the attack
        :param plot_interval: frequency at which printing an extensive plot of a step
        """

        super().__init__(objective_image, objective_mask, debug_root, verbose)

        self.attack_iteration = 0
        self.steps = steps
        self.plot_interval = plot_interval

        self.noise = np.zeros(self.original_objective_image.one_channel().shape)
        self.best_noise = np.zeros(self.original_objective_image.one_channel().shape)

        # create folder to store all the debug deta generated at each step
        os.makedirs(os.path.join(self.debug_folder, "Steps"))

        self.verbose = verbose

        self.visualizer = None

        if step_visualizer:
            self.visualizer = step_visualizer

    def _on_before_attack(self):
        super(BaseIterativeAttack, self)._on_before_attack()
        self.plot_step(self.attacked_image, os.path.join(self.debug_folder, "Steps", str(self.attack_iteration)))

    def attack_one_step(self):
        """
        Perform one step of the attack
        :return:
        """

        self._on_before_attack_step()
        self._attack()
        self._on_after_attack_step()

        return self.attacked_image

    def execute(self):
        """
        Launch the entire attack pipeline
        :return:
        """
        start_time = datetime.now()
        self.write_to_logs("Starting attack pipeline")

        self._on_before_attack()

        for self.attack_iteration in range(self.steps):
            print("### Step: {} ###".format(self.attack_iteration))
            self.attack_one_step()

        self._on_after_attack()

        end_time = datetime.now()
        timedelta = end_time - start_time
        self.write_to_logs("Attack pipeline terminated in {}".format(timedelta))

    def _on_before_attack_step(self):
        """
        Function executed before starting the i-th attack step
        :return:
        """

        self.start_step_time = datetime.now()

    def _on_after_attack_step(self):
        """
        Function executed after ending the i-th attack step
        :return:
        """

        # save the adversarial noise
        np.save(os.path.join(self.debug_folder, 'best-noise.npy'), self.best_noise)

        # log data
        self.end_step_time = datetime.now()
        self.write_to_logs(self._log_step(), False)
        self.attack_iteration += 1

        # generate plots and other visualizations
        if self.plot_interval > 0 and self.attack_iteration % self.plot_interval == 0:
            self.plot_step(self.attacked_image, os.path.join(self.debug_folder, "Steps", str(self.attack_iteration)))

        # save the attacked image
        self.attacked_image.save(os.path.join(self.debug_folder, "attacked image.png"))

    def plot_step(self, image, path):
        """
        Print for debug purposes the state of the attack
        :return:
        """
        if not self.debug_folder or not self.visualizer:
            return

        note = "Step:{}".format(self.attack_iteration)
        self.visualizer.prediction_pipeline(image.to_float(), path,
                                            original_picture=self.original_objective_image.one_channel().to_float(),
                                            note=note,
                                            omask=self.objective_image_mask, debug=False,
                                            adversarial_noise=self.noise)

    def _log_step(self) -> str:
        "Generate the logging to write at each step"
        return " {}) Duration: {}".format(self.attack_iteration, self.end_step_time - self.start_step_time)

    def read_arguments(dataset_root) -> dict:
        """
        Read arguments from the command line or ask for them if they are not present, validate them raising
        an exception if they are invalid, it is called by the launcher script
        :param args: args dictionary containing the arguments passed while launching the program
        :return: kwargs to pass to the attack
        """
        kwarg = BaseAttack.read_arguments(dataset_root)
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", '--steps', default=50, type=int, help='Number of attack steps to perform')
        args = parser.parse_known_args()[0]


        kwarg["steps"] = int(args.steps)
        return kwarg