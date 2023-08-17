import argparse
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment
from Utilities.Experiments.Impilability.ImpilabilityExperiment import ImpilabilityExperiment

attacks = {
    'noiseprint': NoiseprintGlobalIntelligentMimickingAttack,
    'exif': ExifIntelligentAttack,
}

from Utilities.Confs.Configs import Configs

def attack_dataset(dataset, attack, attacked_samples_folder_path):

    configs = Configs("config.yaml", "Attack-ExifAfterNoiseprint-Columbia")

    previous_attack_name = None

    if 'noiseprint' in attacked_samples_folder_path:
        previous_attack_name = 'noiseprint'
    elif 'exif' in attacked_samples_folder_path:
        previous_attack_name = 'exif'

    assert previous_attack_name

    attack_name = f'StackedAttack-{attack.name}-after-{previous_attack_name}'

    # Create folders where to store data about the individual attack executions
    executions_folder_path = configs.create_debug_folder(os.path.join(attack_name, dataset.name, "executions"))

    # Create folder where to store data about the refined outputs of each individual attack
    outputs_folder_path = configs.create_debug_folder(os.path.join(attack_name, dataset.name, "outputs"))

    dataset = DsoDataset(configs["global"]["datasets"]["root"])

    experiment = ImpilabilityExperiment(ExifIntelligentAttack(50, 5, plot_interval=-1, verbosity=0,
                                                         debug_root=executions_folder_path),
                                        outputs_folder_path, dataset,attacked_samples_folder_path)

    experiment.process()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Attack a dataset with the selected attack')

    parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset to attack')
    parser.add_argument('-t', '--target', type=str, help='Name of the detector to target', default=None)

    configs = Configs("config.yaml", f"Attack-{args.target}-{args.dataset}")

    attack = attacks[args.target.lower()]

    datasets = {
        'columbia': ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"]),
        'dso': DsoDataset(configs["global"]["datasets"]["root"]),
    }

    dataset = datasets[args.dataset.lower()]

    assert attack and dataset


