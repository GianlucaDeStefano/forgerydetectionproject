import os

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Ulitities.Experiments.ExifExperiment import ExifExperiment
from Ulitities.Experiments.NoiseprintExperiment import NoiseprintExperiment
from Ulitities.io.folders import create_debug_folder

DEBUG_ROOT = os.path.abspath("./Data/Debug/")
DATASETS_ROOT = os.path.abspath("./Data/Datasets/")
OUTPUT_ROOT = os.path.abspath("./Data/Tampered/DSO/Exif")

root_experiment = create_debug_folder(DATASETS_ROOT)

attack = ExifIntelligentAttack(50, 5, plot_interval=1, verbosity=0,root_debug=root_experiment)

dataset = DsoDatasetDataset(DATASETS_ROOT)

experiment = ExifExperiment(attack,dataset,OUTPUT_ROOT)

experiment.execute()