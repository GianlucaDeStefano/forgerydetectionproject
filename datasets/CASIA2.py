import os
import random
from os.path import basename,splitext
from pathlib import Path
import pandas as pd
from pandas import read_excel
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from datasets.utilityFunctions import getExisting, get_files_with_type

"Class to download the gaia2 dataset"
class CASIA2(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    IMAGE_TYPES = ('*.jpg', '*.tif')
    MASK_TYPES = ('*.png')
    PATH_SHEET_NAME_FIXES = Path(__file__).absolute().parent / "utilities" / "CASIA2_fileNamesCorrection.xlsx"

    def __init__(self,test_proportion:float = 0.1):
        """
        :param test_proportion: percentage of data allocated for testing
        """
        super(CASIA2, self).__init__()
        assert (test_proportion < 1 and test_proportion >= 0)
        self.test_proportion = test_proportion

    def features(self):
        return {
            'image': tfds.features.Tensor(shape=(None,None,3), dtype=tf.float16),
            'mask': tfds.features.Tensor(shape=(None,None,1), dtype=tf.float16),
            'tampered':tfds.features.ClassLabel(num_classes=2)
        }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        return tfds.core.DatasetInfo(
            builder=self,
            description="_DESCRIPTION",
            features=tfds.features.FeaturesDict(self.features()),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # e.g. ('image', 'label')
            homepage='https://dataset-homepage/',
            citation="_CITATION",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""

        datasets_to_download = {
            'samples': 'https://drive.google.com/u/2/uc?id=1YeNjdP3swPSm1ClXJlpOniOyX-9ach9H&export=download',
            'ground_truth': 'https://github.com/namtpham/casia2groundtruth/raw/master/CASIA%202%20Groundtruth.zip',
        }

        #download and extract the datasets
        extracted_ds_paths = dl_manager.download_and_extract(datasets_to_download)

        #get paths of the extracted archives
        self.extracted_path_gt = Path(extracted_ds_paths['ground_truth']) / "CASIA 2 Groundtruth"
        self.extracted_path_data_au = Path(extracted_ds_paths['samples']) / "Au"
        self.extracted_path_data_tp = Path(extracted_ds_paths['samples']) / "Tp"

        #fix filenames in the datasets
        self._fix_files_name(self.extracted_path_data_tp,self.PATH_SHEET_NAME_FIXES)

        #get list of authentic and tapered images
        authentic_files = get_files_with_type(self.extracted_path_data_au,self.IMAGE_TYPES)
        tampered_files = get_files_with_type(self.extracted_path_data_tp,self.IMAGE_TYPES)

        #discard tampered files that have no mask
        tampered_files = list(filter(self._check_mask,tampered_files))

        #shuffle the elements in the 2 lists
        random.shuffle(authentic_files)
        random.shuffle(tampered_files)

        #balance the classes by shortening the authentic and tampered sets to the length of the smallest
        min_len = int(min(len(authentic_files),len(tampered_files)))
        authentic_files = authentic_files[:min_len]
        tampered_files = tampered_files[:min_len]

        #select elements belonging to the train partition
        split_index = int(min_len*(1-self.test_proportion))
        train_authentic = authentic_files[:split_index]
        train_tampered = tampered_files[:split_index]

        # select elements belonging to the test partition
        test_authentic = authentic_files[split_index:]
        test_tampered = tampered_files[split_index:]

        return {"train":self._generate_examples(train_authentic,train_tampered),
                "test":self._generate_examples(test_authentic,test_tampered)}

    def _generate_examples(self, authentic_files : list,tampered_files : list):
        """Generator of examples for each split.
          :param authentic_files: authentic files to include in this set
          :param tampered_files: tampered files to include in this set
        """

        #let's make sure the set is balance
        assert(len(authentic_files) == len(tampered_files))

        for authentic_img in authentic_files:

            # Yields (key, example)
            yield authentic_img, self._process_image(authentic_img)

        for tampered_img in tampered_files:

            # Yields (key, example)
            mask_path = self.extracted_path_gt / (splitext(basename(tampered_img))[0]+"_gt.png")
            yield tampered_img, self._process_image(tampered_img,mask_path)

    def _process_image(self,path,mask_path=None):
        """
        Read the data of an image and the corresponding mask, it the image is authentic, the mask should be black
        :param path: path of the image
        :param mask_path: path of the mask of the image
        :return: n[ array w x h x 3 containing the data of the image and a w x h x 1 np array containing the mask
        """

        image = Image.open(path).convert('RGB')
        #quality = jpeg_quality_of(image)
        image = np.asarray(image)/255
        tampered = 0
        if not mask_path:
            #the image is authentic, the mask has to be completly black
            mask = np.zeros((image.shape[0],image.shape[1],1))
            tampered = 0
        else:
            #if a mask is given the image is tampered, read the ground truth mask
            mask = np.asarray(Image.open(mask_path).convert('1'))[...,np.newaxis]
            tampered = 1

        return {"image":image.astype(np.float16),"mask":mask.astype(np.float16),"tampered":tampered}

    def _fix_files_name(self, path_tampered : Path,path_sheet : Path):
        """The original creator of the dataset made several errors in naming the samples, luckly, on the repo of
            the dataset there is a sheet file containing the right names :
            https://github.com/namtpham/casia2groundtruth/blob/master/Notes/fileNamesCorrection.xlsx

            This function takes that sheet and apply the right names to the files
            :param path_tampered: path of the folder containing tampered images
            :param path_sheep: path of the sheet file containing the fixes
            :param sheet_name: name of the sheet containing the data
        """

        cols = pd.read_excel(path_sheet, header=None, nrows=1).values[0]  # read first row
        df = pd.read_excel(path_sheet, header=None, skiprows=1)  # skip 1 row
        df.columns = cols

        for row in tqdm(df.itertuples(index=True, name='Pandas')):

            old_path = path_tampered / str(row[2])
            new_path = path_tampered / str(row[3])

            if not old_path.is_file():
                continue
            os.rename(old_path, new_path)

    def _check_mask(self,tampered_image:Path) ->bool:
        """
        Check if a mask is available for a given tampered image
        :param tampered_image: tampered image whose mask we hae to find
        :return: Boolean
        """
        mask_path = self.extracted_path_gt / (splitext(basename(tampered_image))[0] + "_gt.png")

        if not mask_path.is_file():
            return  False

        return True