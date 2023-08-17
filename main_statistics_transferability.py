import argparse
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from os.path import join
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.MetricTransferabilityGeneration import MetricGeneratorTransferability

single_attacks_folder_root = '/home/c01gide/CISPA-projects/llm_security_triggers-2023/tesi_archive/alpha_5/risultati/Data/Attacks/SingleAttacks/'

def analyze(experiment_root):
    experiment_name = os.path.basename(os.path.normpath(experiment_root))

    used_dataset_name = "dso" if "dso" in experiment_root.lower() else "columbia"

    used_detector_name = None
    if "retestedon-exif" in experiment_root.lower():
        used_detector_name = 'exif'
    elif "retestedon-noiseprint" in experiment_root.lower():
        used_detector_name = 'noiseprint'
    else:
        raise Exception('Detector not recognized')

    # Find the folder containing the data of the original attack
    original_attack_folder_path = join(single_attacks_folder_root, f"Attack-{used_detector_name}-{used_dataset_name}")

    assert os.path.exists(original_attack_folder_path)

    configs = Configs("config.yaml", 'Statistics-' + experiment_name)

    experiment = MetricGeneratorTransferability(experiment_root, original_attack_folder_path, configs)

    experiment.process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Generate statistics from the results in the selected folder')

    parser.add_argument('-p', '--path', type=str, help='Path to the experiment root')

    args = parser.parse_args()

    analyze(args.path)
