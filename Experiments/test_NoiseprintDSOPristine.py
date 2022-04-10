import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Utilities.Confs.Configs import Configs
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Utilities.Experiments.Attacks.NoiseprintExperiment import NoiseprintExperiment

configs = Configs("../config.yaml", "Noiseprint-DSO-Pristine")

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

attack = NoiseprintGlobalIntelligentMimickingAttack(50, 5, plot_interval=-1, verbosity=0,
                                                    debug_root=configs.create_debug_folder("executions"))
dataset = DsoDatasetDataset(configs["global"]["datasets"]["root"])

experiment = NoiseprintExperiment(attack, dataset, possible_forgery_masks, configs.create_debug_folder("outputs"), True)

experiment.execute()
