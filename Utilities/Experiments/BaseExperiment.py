import os
from abc import abstractmethod
from os.path import basename
from tqdm import tqdm
from Attacks.BaseAttack import BaseAttack
from Datasets.Dataset import Dataset
from Utilities.Image.Picture import Picture
from Utilities.Image.functions import create_target_forgery_map
import statistics as st
import traceback
from Utilities.Logger.Logger import Logger

class ImpossibleToCreateMask(Exception):
    pass

def create_mask(mask,target_forgery_masks):
    target_forgery_mask = None

    for key in target_forgery_masks.keys():
        for i in range(0, 100):
            candidate_mask = create_target_forgery_map(mask.shape,target_forgery_masks[key])

            overlap = (candidate_mask == 1) & (mask == 1)

            if overlap.sum() == 0:
                target_forgery_mask = Picture(candidate_mask)
                break

        if target_forgery_mask is not None:
            logger = Logger()
            logger.logger_module.info(f'Generated mask of size: {key}')
            break

    if target_forgery_mask is None:
        raise ImpossibleToCreateMask("Impossible to create mask")

    return target_forgery_mask


class BaseExperiment(Logger):
    """
    Class used to standardize the experiments to compute metric across different datasets
    """

    def __init__(self, attack: BaseAttack, dataset: Dataset,possible_forgery_masks,debug_root:str,test_authentic):
        self.attack = attack
        self.visualizer = None

        self.dataset = dataset

        self.PSNRs = []

        self.debug_foler = debug_root
        self.output_folder = os.path.join(self.debug_foler,"output")
        self.output_folder_masks = os.path.join(self.debug_foler,"masks")
        os.makedirs(self.output_folder_masks)
        os.makedirs(self.output_folder)

        self.possible_forgery_masks = possible_forgery_masks
        self.test_authentic = test_authentic

    def execute(self):
        """
        Execute the test pipeline
        """
        # foreach image

        images = []
        if self.test_authentic:
            images = self.dataset.get_authentic_images()
        else:
            images = self.dataset.get_forged_images()

        for image_path in tqdm(images):

            if "canonxt_38_sub_05.tif" not in str(image_path):
                print("skipping")
                continue

            print(f"Processing {image_path}")

            # load the image
            try:
                image = Picture(path=image_path)

                # load the mask
                mask, _ = self.dataset.get_mask_of_image(image_path)

                # create a target forgery mask representing the shape of the forgery to copy on the image
                target_forgery_mask = create_mask(mask,self.possible_forgery_masks)

                assert (target_forgery_mask.shape[0] == image.shape[0] and target_forgery_mask.shape[1] == image.shape[
                    1])

                target_forgery_mask.save(os.path.join(self.output_folder_masks,basename(image_path).split(".")[0]+".png"))

                self.attack.setup(image, Picture(mask), target_forgery_mask=target_forgery_mask)
                attacked_image = Picture(self.attack.execute())

                path = os.path.join(self.debug_foler,"tmp.png")
                attacked_image.save(path)

                attacked_image = Picture(path=path)

                self.compute_scores(image,attacked_image, mask, target_forgery_mask)

                self.logger_module.info("PSNR:{}".format(st.mean(self.PSNRs)))
                attacked_image.save(os.path.join(self.output_folder,basename(image_path).split(".")[0]+".png"))
            except KeyboardInterrupt as e:
                exit(0)
            except ImpossibleToCreateMask as e:
                self.logger_module.info(e)
                self.logger_module.warning("Impossible to create mask")
                self.logger_module.warning("Skipping image:{}".format(basename(image_path)))
            except Exception as e:
                self.logger_module.info(traceback.print_exc())
                self.logger_module.info(e)
                self.logger_module.warning("Skipping image:{}".format(basename(image_path)))



    @abstractmethod
    def compute_scores(self,original_image, attacked_image, original_mask, target_mask):
        """
        Compute the f1 and mcc score for the passed attacked image w.r.t the original and target masks
        """
        raise NotImplemented
