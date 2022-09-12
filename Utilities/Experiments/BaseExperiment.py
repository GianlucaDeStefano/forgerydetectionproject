import gc
import traceback
from collections import defaultdict

from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm

from Utilities.Logger.Logger import Logger
from Utilities.Visualizers.BaseVisualizer import BaseVisualizer


class BaseExperiment(Logger):

    def __init__(self, visualizer: BaseVisualizer, debug_root, samples, dataset):
        self.visualizer = visualizer
        self.debug_root = debug_root
        self.samples = samples
        self.dataset = dataset
        self.metrics = {
            "F1-SCORE": f1_score,
            "MCC": matthews_corrcoef
        }

    def process_sample(self, sample_path, mask_path):
        raise NotImplementedError

    def log_result(self):
        """
        Overridable method to customize the statistics that will be generated after each sample is processed
        @return:
        """
        pass

    def log_result_end(self):
        """
        Overridable method to customize the statistics that will be generated after the end of the experimnet
        @return: None
        """
        pass

    def process(self):
        self.pristine_scores = defaultdict(lambda: [])

        try:

            for i,sample_path in enumerate(tqdm(self.samples)):
                self.logger_module.info(f"Processing sample {i}/{len(self.samples)}: {sample_path}")
                gt_mask, _ = self.dataset.get_mask_of_image(sample_path)
                self.process_sample(sample_path, gt_mask)

                # print the current statistics in the log file
                self.log_result()
                gc.collect()

            # Print the final statistics
            self.log_result_end()

        except Exception as e:
            self.logger_module.error(f"EXCEPTION")
            self.logger_module.error(traceback.format_exc())
            self.logger_module.error(e)

