import argparse
import os

from Attacks.Exif.Lots.Lots4Exif_original import Lots4ExifOriginal
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Attacks.LotsExperiment import LotsExperiment

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=False, default=None)
args = parser.parse_known_args()[0]

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml", "LOTS-attack-exif-Columbia")

dataset = ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"])
attack = Lots4ExifOriginal(50, 5, plot_interval=-1)
samples = [sample_path for sample_path in dataset.get_forged_images()]

experiment = LotsExperiment(attack, debug_root=configs.create_debug_folder("outputs"), samples=samples, dataset=dataset)

experiment.process()
