import os
import traceback

from Attacks.Exif.Mimicking.ExifMimickingIntelligentAttack import ExifIntelligentAttack
from Attacks.Noiseprint.Mimiking.NoiseprintMimickingIntelligentGlobal import NoiseprintGlobalIntelligentMimickingAttack
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDataset
from Utilities.Confs.Configs import Configs
from Utilities.Experiments.Attacks.MimicryExperiment import MimicryExperiment

configs = Configs("config.yaml", "Mimicry attacks")

datasets = [
    ColumbiaUncompressedDataset(configs["global"]["datasets"]["root"]),
    DsoDataset(configs["global"]["datasets"]["root"]),
]

attacks = [NoiseprintGlobalIntelligentMimickingAttack, ExifIntelligentAttack]

for attack in attacks:

    for dataset in datasets:

        try:

            # Create folders where to store data about the individual attack executions
            executions_folder_path = configs.create_debug_folder(os.path.join(attack.name, dataset.name, "executions"))

            # Create folder where to store data about the refined outputs of each individual attack
            outputs_folder_path = configs.create_debug_folder(os.path.join(attack.name, dataset.name, "outputs"))

            # get a list of all the samples in a dataset
            samples = [sample_path for sample_path in dataset.get_forged_images()]

            # instance the experiment to run
            experiment = MimicryExperiment(
                attack(50, 5, plot_interval=-1, verbosity=0, debug_root=executions_folder_path), outputs_folder_path,
                samples, dataset)

            # run the experiment
            experiment.process()

        except Exception as e:
            traceback.print_exc()
