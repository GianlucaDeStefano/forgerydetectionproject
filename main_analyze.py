import os
import traceback
from Utilities.Experiments.AnalyzeSamples2 import AnalyzeSamples
from Utilities.Confs.Configs import Configs

"""
    This script is used to analyze a set of attack results all contained in distinct folders inside a root
"""


experiment_root = "/home/gianluca/RISULTATI/DECOY/"

for experiment_path in [ f.path for f in os.scandir(experiment_root) if f.is_dir() ][5:6]:

    for execution_path in [ f.path for f in os.scandir(experiment_path) if f.is_dir() ]:

        if experiment_path == execution_path:
            continue

        if "attack" not in execution_path.lower():
            continue

        experiment_name = os.path.basename(os.path.normpath(experiment_path))

        configs = Configs("config.yaml", 'Analysis-Samples-' + experiment_name)

        try:

            experiment = AnalyzeSamples(execution_path, configs)

            experiment.process()
        except Exception as e:
            print(e)
            traceback.print_exc()
        break
