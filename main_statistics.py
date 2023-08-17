import argparse
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from Utilities.Experiments.MetricGeneration import MetricGenerator
from Utilities.Confs.Configs import Configs


def analyze(experiment_root):
    experiment_name = os.path.basename(os.path.normpath(experiment_root))

    configs = Configs("config.yaml", 'Statistics-' + experiment_name)

    experiment = MetricGenerator(experiment_root, configs)

    experiment.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Generate statistics from the results in the selected folder')

    parser.add_argument('-p', '--path', type=str, help='Path to the experiment root')

    args = parser.parse_args()

    analyze(args.path)
