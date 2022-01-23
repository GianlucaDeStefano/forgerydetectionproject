from os.path import basename
from tqdm import tqdm
from Attacks.BaseAttack import BaseAttack
from Datasets.Dataset import Dataset
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import create_target_forgery_map
import statistics as st

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


class Experiment:
    """
    Class used to standardize the experiments to compute metric across different datasets
    """

    def __init__(self, attack: BaseAttack, dataset: Dataset):
        self.attack = attack
        self.detector = self.attack.detector

        self.dataset = dataset

        self.PSNR = []

        self.f1_original_forgery = []
        self.f1_target_forgery = []

        self.mcc_original_forgery = []
        self.mcc_final_forgery = []

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

                self.compute_scores(attacked_image,mask,target_forgery_mask)

                print("f1_original:{:.2} mcc_original:{:.2}".format(st.mean(self.f1_original_forgery),st.mean(self.mcc_original_forgery)))
                print("f1_target:{} mcc_target:{}".format(st.mean(self.f1_original_forgery),st.mean(self.mcc_target_forgery)))

            except Exception as e:
                print("ERROR:", e)
                print("Skipping image:{}".format(basename(image_path)))

    def compute_scores(self, attacked_image, original_mask, target_mask):
        """
        Compute the f1 and mcc score for the passed attacked image w.r.t the original and target masks
        """
