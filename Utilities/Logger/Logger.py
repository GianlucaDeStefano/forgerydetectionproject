import logging
import os
import sys
from os.path import join


def setup_logging(root_folder):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    # In the console print only messages having a level equal or above console_debug_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter(fmt="%(levelname)-8s - %(message)s", datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Log all data into the console.log file
    file_handler = logging.FileHandler(filename=os.path.join(root_folder, "console.log"), mode="a")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                  datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class Logger:
    """
    This class implements a usefull interface to add logging functionalities easily to any of its children
    """

    @property
    def logger_module(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)
