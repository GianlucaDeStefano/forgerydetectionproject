import argparse
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from os.path import join

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment
from Utilities.Experiments.Impilability.ImpilabilityExperiment import ImpilabilityExperiment
from Utilities.Experiments.MetricGeneration import MetricGenerator
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.MetricTransferabilityGeneration import MetricGeneratorTransferability

experiment_root = "/home/c01gide/CISPA-home/tesi/Data/SampleAnalysis/Mimicry"
experiment_root_transferability = "/home/c01gide/CISPA-home/tesi/Data/SampleAnalysis/Transferability"

for experiment_path in [f.path for f in os.scandir(experiment_root_transferability) if f.is_dir()]:

    for execution_path in [f.path for f in os.scandir(experiment_path) if f.is_dir()]:

        if experiment_path == execution_path:
            continue

        experiment_name = os.path.basename(os.path.normpath(experiment_path))

        used_dataset_name = "DSO" if "DSO" in experiment_path else "Columbia"

        used_detector_name = "Exif" if "retested on Exif" in experiment_path else "Noiseprint"

        pristine_data_experiment_path = join(experiment_root, f"Attack-{used_detector_name}-{used_dataset_name}")

        experiments_dirs = [name for name in os.listdir(pristine_data_experiment_path) if os.path.isdir(join(pristine_data_experiment_path,name))]

        assert (len(experiments_dirs) == 1)

        pristine_data_experiment_path = join(pristine_data_experiment_path,experiments_dirs[0])

        configs = Configs("config.yaml", 'Statistics-' + experiment_name)

        try:

            experiment = MetricGeneratorTransferability(execution_path, pristine_data_experiment_path, configs)

            experiment.process()
        except Exception as e:
            print(e)
        break
