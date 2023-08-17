import argparse
import traceback
import os

from Utilities.Experiments.Impilability.ImpilabilityExperiment import ImpilabilityExperiment

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment

attacks = {
    'noiseprint': NoiseprintGlobalIntelligentMimickingAttack,
    'exif': ExifIntelligentAttack,
}


def attack_dataset(args):

    # Retrieve the instance of the attack to execute
    attack = attacks[args.target.lower()]

    # Get the folder of 'pre-attacked' samples to 'imprint' a second attack on
    pre_attacked_folder = args.folder_attacked_samples

    # define experiment's name
    attack_name = f"Attack-{args.target}-{args.dataset}"

    # If this attack is 'piling' over another
    if pre_attacked_folder:

        previous_attack_name = None

        assert not ('exif' in pre_attacked_folder and 'noiseprint' in pre_attacked_folder)

        if 'noiseprint' in pre_attacked_folder:
            previous_attack_name = 'noiseprint'
        elif 'exif' in pre_attacked_folder:
            previous_attack_name = 'exif'

        assert previous_attack_name

        attack_name = f'Stacked-{args.target}after{previous_attack_name}-{args.dataset}'

    configs = Configs("config.yaml", attack_name)

    datasets = {
        'columbia': ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"]),
        'dso': DsoDataset(configs["global"]["datasets"]["root"]),
    }

    dataset = datasets[args.dataset.lower()]

    assert attack and dataset

    # Create folders where to store data about the individual attack executions
    executions_folder_path = configs.create_debug_folder(os.path.join(attack_name, dataset.name, "executions"))

    # Create folder where to store data about the refined outputs of each individual attack
    outputs_folder_path = configs.create_debug_folder(os.path.join(attack_name, dataset.name, "outputs"))

    if not pre_attacked_folder:

        # get a list of all the samples in a dataset
        samples = [sample_path for sample_path in dataset.get_forged_images()]

        # instance the experiment to run
        experiment = MimicryExperiment(
            attack(50, 5, plot_interval=-1, verbosity=0, debug_root=executions_folder_path), outputs_folder_path,
            samples, dataset)
    else:

        configs.logger_module.info(f'Stacking the attack on top of samples in: {pre_attacked_folder}')

        # instance the experiment to run
        experiment = ImpilabilityExperiment(attack(50, 5, plot_interval=-1, verbosity=0, debug_root=executions_folder_path),
                                            outputs_folder_path, dataset,
                                            pre_attacked_folder)

    # run the experiment
    experiment.process()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Attack a dataset with the selected attack')

    parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset to attack')
    parser.add_argument('-t', '--target', type=str, help='Name of the detector to target')
    parser.add_argument('-f', '--folder_attacked_samples', type=str, help='Path of the folder containing pre-attacked samples', default=None)

    args = parser.parse_args()

    attack_dataset(args)
