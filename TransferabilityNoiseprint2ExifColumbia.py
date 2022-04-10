import tensorflow as tf
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Utilities.Experiments.Transferability.ExifTransferabilityExperiment import ExifTransferabilityExperiment
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Utilities.Confs.Configs import Configs

configs = Configs("config.yaml", "Noiseprint2Exif-Columbia")

data_path = "/home/gianlucadestefano/Deep-detectors-attacks/Data/Debug/Noiseprint-Columbia/1644260423.4584503/outputs"

original_dataset = ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"])

experiment = ExifTransferabilityExperiment(configs.create_debug_folder("results"), data_path, original_dataset)

experiment.execute()
