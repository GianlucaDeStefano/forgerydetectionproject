import argparse
import os
from Utilities.Experiments.MetricGeneration import MetricGenerator
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.MetricGenerationLots import MetricGeneratorLots

experiment_root = "/home/gianluca/RISULTATI/LOTS"

for experiment_path in [ f.path for f in os.scandir(experiment_root) if f.is_dir() ]:

    for execution_path in [ f.path for f in os.scandir(experiment_path) if f.is_dir() ]:

        if experiment_path == execution_path:
            continue

        experiment_name = os.path.basename(os.path.normpath(experiment_path))

        if 'noiseprint' not in experiment_name:

            configs = Configs("config.yaml", 'Statistics-' + experiment_name)

            try:

                experiment = MetricGeneratorLots(execution_path, configs)

                experiment.process()
            except Exception as e:
                print(e)
            break
