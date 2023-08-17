import datetime
import json
import logging
import os
import sys
import time
from os.path import abspath, join
from pathlib import Path

import yaml
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Utilities.Logger.Logger import Logger, setup_logging


class Configs(Logger):
    """
    This class manages the loading and reading of the configs form the passed file.
    It also creates the needed folders for the saving of logs and results.
    """

    def __init__(self, config_file_path: str, category="generic"):
        """
        :param config_file_path: path to config to the file path to load :param category: category to assign to the
        run, used to differentiate between executions of various script and save the respective results into
        different folders
        """
        self.config_file = config_file_path
        # Read YAML file
        with open(self.config_file, 'r') as stream:
            self._config_dict = yaml.safe_load(stream)

        self.category = category
        self.timestamp = time.time()

        setup_logging(self.debug_root)

        # set up usefull environment variables
        os.environ["DEBUG-ROOT"] = self.debug_root
        os.environ["CACHE"] = self.create_debug_folder("cache")

        # log current time in the log file
        self.logger_module.info(f"Starting execution at: {datetime.time()}")

        # Write in the logs the config that we are using
        self.logger_module.info("Configs:")
        self.logger_module.info(f"Debug root folder: {self.debug_root}")
        self.logger_module.info(json.dumps(self._config_dict, sort_keys=True, indent=4))

    @property
    def debug_root(self):
        """
        :return: Returns the path to the folder created to contain the data generated during this run
        """
        folder_path = abspath(join(self._config_dict["global"]["debug"]["path"], self.category))

        if not Path(folder_path).exists():
            os.makedirs(folder_path)

        return folder_path

    def create_debug_folder(self,path_to_append):
        """
        Using the debug foolder as root of the path, crate into it the specified folders
        :param path_to_append: folders to add into the debug root folder
        :return: absolute path to the created folders
        """

        desired_path = join(self.debug_root,path_to_append)

        if not Path(desired_path).exists():
            os.makedirs(desired_path)

        return desired_path

    def __getitem__(self, key):
        """
        Wrapper of the __getitem__ method of the _config_dict to allow for an easy reading of the settings
        :param key:
        :return:
        """
        val = self._config_dict.__getitem__(key)
        return val