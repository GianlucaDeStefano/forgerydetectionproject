import argparse
import traceback
import os

import numpy as np

from Datasets.Dataset import resize_mask
from Utilities.Experiments.Impilability.ImpilabilityExperiment import ImpilabilityExperiment
from Utilities.Image.Picture import Picture

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

def attack_image(args):

    assert args.image

    # define experiment's name
    attack_name = f"Attack-{args.detector}-{os.path.basename(args.image)}"

    configs = Configs("config.yaml", attack_name)

    output_path = configs.create_debug_folder("outputs")

    print(f'Output path: {output_path}')

    original_forgery_mask = Picture(path=args.original_mask)
    original_forgery_mask.save(f'{output_path}/original_forgery_mask.png')
    original_forgery_mask = np.rint(Picture(original_forgery_mask[:, :, 0]) / 255)

    target_forgery_mask = Picture(path=args.target_mask)
    target_forgery_mask.save(f'{output_path}/target_forgery_mask.png')
    target_forgery_mask = np.rint(Picture(target_forgery_mask[:, :, 0]) / 255)

    # Retrieve the instance of the attack to execute
    attack = attacks[args.detector.lower()](50, args.strength, plot_interval=-1, verbosity=0, debug_root=output_path)

    # setup the attack
    attack.setup(args.image, original_forgery_mask, args.image, original_forgery_mask,
                      target_forgery_mask)

    attack.visualizer.save_prediction_pipeline(os.path.join(output_path, 'pristine_prediction.png'), original_forgery_mask)

    # compute the pristine heatmap
    heatmap_pristine = attack.visualizer.metadata["heatmap"]

    # save pristine heatmap
    pristine_heatmap_path = os.path.join(output_path, "pristine_heatmap.npy")
    np.save(pristine_heatmap_path, np.array(heatmap_pristine))

    # execute the attack
    _, attacked_sample = attack.execute()

    # save the attacked sample in the file system
    attacked_sample_path = os.path.join(output_path, f'attacked {os.path.basename(args.image)}')
    attacked_sample.save(attacked_sample_path)

    # compute the attacked heatmap
    attack.visualizer.process_sample(attacked_sample_path)
    heatmap_attacked = attack.visualizer.metadata["heatmap"]

    # save attacked heatmap
    attacked_heatmap_path = os.path.join(output_path, "attacked_heatmap.npy")
    np.save(attacked_heatmap_path, np.array(heatmap_attacked))

    # save result of the detector on the attacked image
    attack.visualizer.save_prediction_pipeline(os.path.join(output_path, 'attacked_prediction.png'),
                                             target_forgery_mask)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Attack a dataset with the selected attack')

    parser.add_argument('-i', '--image', type=str, help='Path to the image to attack', default=None)
    parser.add_argument('-o', '--original_mask', type=str, help='Path to the original forgery mask to attack', default=None)
    parser.add_argument('-t', '--target_mask', type=str, help='Path to the target forgery mask to attack', default=None)

    parser.add_argument('-d', '--detector', type=str, help='Name of the detector to target')
    parser.add_argument('-s', '--strength', type=float, help='Strength to apply to the attack', default=5.0)


    args = parser.parse_args()

    attack_image(args)
