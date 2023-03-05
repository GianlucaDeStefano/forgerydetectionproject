import argparse
import os

from Attacks.Exif.Lots.Lots4Exif_original import Lots4ExifOriginal
from Attacks.Noiseprint.Lots.Lots4Noiseprint_globalmap import Lots4NoiseprintAttackGlobalMap
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Attacks.LotsExperiment import LotsExperiment

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=False, default=None)
args = parser.parse_known_args()[0]

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml", "LOTS attacks")


attacks = [Lots4NoiseprintAttackGlobalMap]


datasets = [
    ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"]),
    DsoDataset(configs["global"]["datasets"]["root"]),
]

for attack in attacks:

    for dataset in datasets:

        # Create folders where to store data about the individual attack executions
        executions_folder_path = configs.create_debug_folder(os.path.join(attack.name, dataset.name, "executions"))

        # Create folder where to store data about the refined outputs of each individual attack
        outputs_folder_path = configs.create_debug_folder(os.path.join(attack.name, dataset.name, "outputs"))

        # get a list of all the samples in a dataset
        samples = [sample_path for sample_path in dataset.get_forged_images()]

        # instance the experiment to run
        experiment = LotsExperiment(
            attack(50, 5, plot_interval=-1, verbosity=0, debug_root=executions_folder_path), outputs_folder_path,
            samples, dataset)
        experiment.process()
