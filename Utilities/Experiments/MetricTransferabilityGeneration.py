from os.path import join

from Utilities.Experiments.MetricGeneration import MetricGenerator


class MetricGeneratorTransferability(MetricGenerator):

    def __init__(self, attacked_samples_root, pristine_samples_root, configs, dataset=None):
        """
        Class used to generate statistics on the transferability of the attacks
        @param attacked_samples_root: folder containing the attacked samples data
        @param pristine_samples_root: folder containing the data of the samples tested without any attack
        """

        super().__init__(attacked_samples_root, configs, dataset)

        self.attacked_samples_data_root = attacked_samples_root
        self.pristine_samples_data_root = pristine_samples_root

    @property
    def attacked_samples_folder(self):
        return join(self.attacked_samples_data_root, "outputs", "attackedSamples")

    @property
    def attacked_samples_heatmaps_folder(self):
        return join(self.attacked_samples_data_root, "outputs", "pristine", "heatmaps")

    @property
    def attacked_samples_target_forgery_masks_folder(self):
        return join(self.attacked_samples_folder, "targetForgeryMasks")

    @property
    def pristine_samples_folder(self):
        return join(self.pristine_samples_data_root, "outputs", "pristine")

    @property
    def pristine_samples_heatmaps_folder(self):
        return join(self.pristine_samples_folder, "heatmaps")

    def get_samples_to_process(self):
        f = self.dataset.get_forged_images()

        return [str(p) for p in f]