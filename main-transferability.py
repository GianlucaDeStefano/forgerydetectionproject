import argparse
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
# from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment
from Utilities.Experiments.Impilability.ImpilabilityExperiment import ImpilabilityExperiment
from Utilities.Experiments.Transferability.TransferabilityExperiment import TransferabilityExperiment
# from Utilities.Visualizers.ExifVisualizer import ExifVisualizer
# from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer
from Utilities.Visualizers.ExifVisualizer import ExifVisualizer
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=False, default=None)
args = parser.parse_known_args()[0]

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml", "Attack-ExifAfterNoiseprint-DSO retested on Noiseprint")

dataset = ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"])

experiment = TransferabilityExperiment(NoiseprintVisualizer(),debug_root=configs.create_debug_folder("outputs"),dataset=dataset,
                                       attacked_samples_folder_path="/home/c01gide/CISPA-home/tesi/Data/SampleAnalysis/Attack-ExifAfterNoiseprint-DSO/1677914982.4525623/outputs/attackedSamples")

experiment.process()
