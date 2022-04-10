from Utilities.Confs.Configs import Configs
from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Utilities.Experiments.Attacks.ExifExperiment import ExifExperiment
from Utilities.io.folders import create_debug_folder

configs = Configs("../config.yaml", "Exif-DSO")

possible_forgery_masks = {
    "300x300": [
        "./Data/custom/target_forgery_masks/DSO/1.png",
        "./Data/custom/target_forgery_masks/DSO/2.png",
        "./Data/custom/target_forgery_masks/DSO/3.png"
    ],
    "150x150": [
        "./Data/custom/target_forgery_masks/COLUMBIA/1.png",
        "./Data/custom/target_forgery_masks/COLUMBIA/2.png",
        "./Data/custom/target_forgery_masks/COLUMBIA/3.png"
    ]
}

root_experiment = configs.debug_root

dataset = DsoDatasetDataset(configs["global"]["datasets"]["root"])

experiment = ExifExperiment(None, dataset, possible_forgery_masks, configs.create_debug_folder("outputs"), False)

experiment.execute()
