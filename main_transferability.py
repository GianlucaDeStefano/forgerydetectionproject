import argparse
import os

from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Utilities.Visualizers.ExifVisualizer import ExifVisualizer

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Experiments.Transferability.TransferabilityExperiment import TransferabilityExperiment
from Utilities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer

from Utilities.Confs.Configs import Configs

visualizers = {
    'noiseprint': NoiseprintVisualizer,
    'exif':ExifVisualizer
}

def generate_transferability_samples(args):

    experiment_name = os.path.basename(args.parent_experiment_folder)

    samples_folder_path = f'{args.parent_experiment_folder}/outputs/attackedSamples'

    configs = Configs("config.yaml", f'Transferability-{experiment_name}-retestedOn-{args.target}')

    datasets = {
        'columbia': ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"]),
        'dso': DsoDataset(configs["global"]["datasets"]["root"]),
    }

    dataset = datasets[args.dataset.lower()]

    visualizer = visualizers[args.target]()

    experiment = TransferabilityExperiment(visualizer,debug_root=configs.create_debug_folder("outputs"),dataset=dataset,
                                           attacked_samples_folder_path=samples_folder_path)

    experiment.process()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Attack a dataset with the selected attack')

    parser.add_argument('-d', '--dataset', type=str, help='Name of the dataset to attack')
    parser.add_argument('-t', '--target', type=str, help='Name of the detector to target')
    parser.add_argument('-f', '--parent_experiment_folder', type=str, help='Path of the folder containing pre-attacked samples', default=None)

    args = parser.parse_args()

    generate_transferability_samples(args)