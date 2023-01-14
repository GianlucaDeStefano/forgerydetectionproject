import argparse
import os

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--gpu', required=False, default=None)
args = parser.parse_known_args()[0]

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from Utilities.Confs.Configs import Configs

dataset_name = args.dataset
configs = Configs("config.yaml", f"Attack-Noiseprint-{dataset_name}")

if dataset_name == 'DSO':
    dataset = DsoDataset(configs["global"]["datasets"]["root"])
elif dataset_name == 'Columbia':
    dataset = ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"])
else:
    raise Exception('Unknown Dataset')

samples = [sample_path for sample_path in dataset.get_forged_images()]

experiment = MimicryExperiment(ExifIntelligentAttack(50, 5, plot_interval=-1, verbosity=0,
                                                     debug_root=configs.create_debug_folder("executions"),
                                                     batch_size=512),
                               configs.create_debug_folder("outputs"), samples, dataset)

experiment.process()
