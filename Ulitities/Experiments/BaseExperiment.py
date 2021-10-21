import os
from abc import abstractmethod
from os.path import basename
from tqdm import tqdm
from Attacks.BaseAttack import BaseAttack
from Datasets.Dataset import Dataset
from Ulitities.Image.Picture import Picture
from Ulitities.Image.functions import create_target_forgery_map
import statistics as st
import traceback

from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer


def create_mask(mask):
    target_forgery_mask = None
    for i in range(0, 20):
        candidate_mask = create_target_forgery_map(mask.shape)

        overlap = (candidate_mask == 1) & (mask == 1)

        if overlap.sum() == 0:
            target_forgery_mask = Picture(candidate_mask)
            break

    if target_forgery_mask is None:
        raise Exception("Impossible to create mask")

    return target_forgery_mask


class BaseExperiment:
    """
    Class used to standardize the experiments to compute metric across different datasets
    """

    def __init__(self, attack: BaseAttack, dataset: Dataset,debug_root:str):
        self.attack = attack
        self.visualizer = None

        self.dataset = dataset

        self.PSNR = []

        self.f1_original_forgery = []
        self.f1_target_forgery = []

        self.mcc_original_forgery = []
        self.mcc_target_forgery = []

        self.PSNRs = []

        self.debug_foler = debug_root

    def execute(self):
        """
        Execute the test pipeline
        """
        # foreach image
        for image_path in tqdm(self.dataset.get_forged_images()):

            # load the image
            image = Picture(path=image_path)

            # load the mask
            mask, _ = self.dataset.get_mask_of_image(image_path)

            # create a target forgery mask representing the shape of the forgery to copy on the image
            target_forgery_mask = create_mask(mask)

            try:

                assert (target_forgery_mask.shape[0] == image.shape[0] and target_forgery_mask.shape[1] == image.shape[
                    1])

                self.attack.setup(image, Picture(mask), target_forgery_mask=target_forgery_mask)
                attacked_image = Picture(self.attack.execute())

                path = os.path.join(self.debug_foler,"tmp.png")
                attacked_image.save(path)

                attacked_image = Picture(path=path)

                self.compute_scores(image,attacked_image, mask, target_forgery_mask)

                print("f1_original:{:.2} mcc_original:{:.2}".format(st.mean(self.f1_original_forgery),
                                                                    st.mean(self.mcc_original_forgery)),flush=True)

                print("f1_target:{} mcc_target:{}".format(st.mean(self.f1_target_forgery),
                                                          st.mean(self.mcc_target_forgery)), flush=True)

                print("PSNR:{}".format(st.mean(self.PSNRs)),flush=True)

            except Exception as e:
                traceback.print_exc()
                print("Skipping image:{}".format(basename(image_path)))

    @abstractmethod
    def compute_scores(self,original_image, attacked_image, original_mask, target_mask):
        """
        Compute the f1 and mcc score for the passed attacked image w.r.t the original and target masks
        """
        raise NotImplemented
