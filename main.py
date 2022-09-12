import argparse
import os

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment
from Utilities.Experiments.Impilability.ImpilabilityExperiment import ImpilabilityExperiment

# from Utilities.Experiments.Transferability.TransferabilityExperiment import TransferabilityExperiment
# from Utilities.Visualizers.ExifVisualizer import ExifVisualizer
# from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=False, default=None)
args = parser.parse_known_args()[0]

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml", "Attack-Noiseprint-DSO")

dataset = DsoDataset(configs["global"]["datasets"]["root"])

samples = [sample_path for sample_path in dataset.get_forged_images()]

experiment = MimicryExperiment(NoiseprintGlobalIntelligentMimickingAttack(50, 5, plot_interval=-1, verbosity=0,
                                                     debug_root=configs.create_debug_folder("executions")),
                                    configs.create_debug_folder("outputs"), samples, dataset)

experiment.process()



