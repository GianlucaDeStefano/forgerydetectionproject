import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Transferability.TransferabilityExperiment import TransferabilityExperiment
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer

from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml", "Attack-ExifAfterNoiseprint-DSO retested on Noiseprint")

dataset = DsoDataset(configs["global"]["datasets"]["root"])

experiment = TransferabilityExperiment(NoiseprintVisualizer(),debug_root=configs.create_debug_folder("outputs"),dataset=dataset,
                                       attacked_samples_folder_path="/home/c01gide/CISPA-home/tesi/Data/SampleAnalysis/Attack-ExifAfterNoiseprint-DSO/1677914982.4525623/outputs/attackedSamples")

experiment.process()
