import tensorflow as tf

from Utilities.Experiments.Transferability.ExifTransferabilityExperiment import ExifTransferabilityExperiment

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.Transferability.NoiseprintTransferabilityExperiment import \
    NoiseprintTransferabilityExperiment

configs = Configs("config.yaml", "Noiseprint2Exif-DSO")

data_path = "/home/gianlucadestefano/Deep-detectors-attacks/Data/Debug/Noiseprint-DSO/1646385421.243167/outputs"

original_dataset = DsoDatasetDataset(configs["global"]["datasets"]["root"])

experiment = ExifTransferabilityExperiment(configs.create_debug_folder("results"), data_path, original_dataset)

experiment.execute()
