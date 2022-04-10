import tensorflow as tf

from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.Transferability.NoiseprintTransferabilityExperiment import \
    NoiseprintTransferabilityExperiment

configs = Configs("config.yaml", "Exif2Noiseprint-Columbia")

data_path = "/home/gianlucadestefano/Deep-detectors-attacks/Data/Debug/Exif-Columbia/1645602457.1072206/outputs"

original_dataset = ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"])

experiment = NoiseprintTransferabilityExperiment(configs.create_debug_folder("results"), data_path, original_dataset)

experiment.execute()
