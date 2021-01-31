import os
import random
from os.path import basename, splitext
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
from tqdm import tqdm
import tensorflow as tf

from Datasets.Modifications.BlurModification import BlurModification
from Datasets.Modifications.CompressionModification import CompressionModification
from Datasets.Modifications.ExposureModification import ExposureModification
from Datasets.Modifications.GaussianModification import GaussianModification
from Datasets.Modifications.PoissonModification import PoissonModification
from Datasets.Modifications.SaltAndPepperModification import SaltAndPepperModification
from Datasets.Modifications.SpeckleModification import SpeckleModification
from Datasets.Utilities.Maps.Noiseprint.NoiseprintExtractor import NoiseprintExtractor
from Datasets.Utilities.Maps.Noiseprint.noiseprint import gen_noiseprint
from Datasets.Utilities.Maps.SRM.SRMExtractor import SRMExtractor
from Datasets.Utilities.utilityFunctions import get_files_with_type

"Class to download the CASIA2 dataset"


class CASIA2(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    IMAGE_TYPES = ('*.jpg', '*.tif')
    MASK_TYPES = ('*.png')
    PATH_SHEET_NAME_FIXES = Path(__file__).absolute().parent / "Utilities" / "CASIA2_fileNamesCorrection.xlsx"

    test_proportion = 0.01

    def __init__(self, **kwargs):
        """
        :param test_proportion: percentage of data allocated for testing
        """

        # call the parent's __init__
        super(CASIA2, self).__init__()

        # check if the test_proportion parameter is in a valid range
        test_proportion = kwargs.get("test_proportion", 0.1)
        assert (test_proportion < 1 and test_proportion >= 0)

        # check if the validation_proportion parameter is in a valid range
        validation_proportion = kwargs.get("validation_proportion", 0.1)
        assert (validation_proportion < 1 and validation_proportion >= 0)

        # check if testporportion and validation proportion together are still in a valid range
        assert (validation_proportion + test_proportion < 1)

        self.test_proportion = test_proportion
        self.validation_proportion = validation_proportion

        # if true prints logs while building the dataset
        self.verbose = kwargs.get("verbose", True)

        # the shape of the images accepted by the model
        self.supported_shape = (256, 384, 3)

        # setup data generators
        self.SRMGenerator = SRMExtractor()
        self.NoiseprintGenerator = NoiseprintExtractor()

    def features(self):
        """
        This function return a dictionary holding the type and shape of the data this builder object
        generates foreach sample
        :return:
        """

        desired_shape_3 = (256, 384, 3)
        desired_shape_1 = (256, 384, 1)

        return {
            'image': tfds.features.Image(shape=desired_shape_3),
            'noiseprint': tfds.features.Tensor(shape=desired_shape_1, dtype=tf.float32),
            'SRM': tfds.features.Image(shape=desired_shape_3),
            'flipped': tfds.features.ClassLabel(num_classes=2),
            'tampered': tfds.features.ClassLabel(num_classes=2)
        }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata describint the dataset"""

        return tfds.core.DatasetInfo(
            builder=self,
            description="_DESCRIPTION",
            features=tfds.features.FeaturesDict(self.features()),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "tampered"),
            homepage='https://dataset-homepage/',
            citation="_CITATION",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Download the data and define splits."""

        # list of the Datasets to download, one contains the training samples, the other contains the ground truths
        datasets_to_download = {
            'samples': 'https://drive.google.com/u/2/uc?id=1YeNjdP3swPSm1ClXJlpOniOyX-9ach9H&export=download',
            'ground_truth': 'https://github.com/namtpham/casia2groundtruth/raw/master/CASIA%202%20Groundtruth.zip',
        }

        # download and extract the Datasets
        extracted_ds_paths = dl_manager.download_and_extract(datasets_to_download)

        # get paths of the extracted archives
        self.extracted_path_gt = Path(extracted_ds_paths['ground_truth']) / "CASIA 2 Groundtruth"
        self.extracted_path_data_au = Path(extracted_ds_paths['samples']) / "Au"
        self.extracted_path_data_tp = Path(extracted_ds_paths['samples']) / "Tp"

        # fix filenames in the Datasets
        self._fix_files_name(self.extracted_path_data_tp, self.PATH_SHEET_NAME_FIXES)

        # get list of authentic and tapered images
        authentic_files = get_files_with_type(self.extracted_path_data_au, self.IMAGE_TYPES)
        tampered_files = get_files_with_type(self.extracted_path_data_tp, self.IMAGE_TYPES)

        # discard tampered files that have no mask
        tampered_files = list(filter(self._check_mask, tampered_files))

        # discard invalid files
        authentic_files = self._keep_valid(authentic_files)
        tampered_files = self._keep_valid(tampered_files)

        print("Found {} pristine and {} tampered images".format(len(authentic_files),len(tampered_files)))

        authentic_files = authentic_files[:100]

        # shuffle the elements in the 2 lists
        random.shuffle(authentic_files)
        random.shuffle(tampered_files)

        # balance the classes by shortening the authentic and tampered sets to the length of the smallest
        min_len = int(min(len(authentic_files), len(tampered_files)))
        authentic_files = authentic_files[:min_len]
        tampered_files = tampered_files[:min_len]

        # select elements belonging to the train partition
        split_index = int(min_len * (1 - self.test_proportion - self.validation_proportion))
        train_authentic = authentic_files[:split_index]
        train_tampered = tampered_files[:split_index]

        # select elements belonging to the validation partition
        split_index_val = int(min_len * (1 - self.test_proportion))
        val_authentic = authentic_files[split_index:split_index_val]
        val_tampered = tampered_files[split_index:split_index_val]

        # select elements belonging to the test partition
        test_authentic = authentic_files[split_index_val:]
        test_tampered = tampered_files[split_index_val:]

        #create Modification classes

        #Blurred images
        blur_5 = BlurModification(5)
        blur_7 = BlurModification(7)

        #Salt and pepper
        salt = SaltAndPepperModification(0.5,0.004)

        # Change exposure
        exposed = ExposureModification(50)

        #JPEG Compression
        compressed = CompressionModification(5)

        #speckle
        #speckle = SpeckleModification()

        #gaussian
        #gaussian = GaussianModification()

        #Poisson
        #poisson = PoissonModification()


        return {#"train": self._generate_examples("train",train_authentic, train_tampered, []),
                #"validation": self._generate_examples("validation",val_authentic, val_tampered, []),
                #"test": self._generate_examples("test",test_authentic, test_tampered, []),
                #"test_blur_5": self._generate_examples("test", test_authentic, test_tampered, [blur_5]),
                #"test_blur_7": self._generate_examples("test",test_authentic, test_tampered, [blur_7]),
                #"test_salt": self._generate_examples("test", test_authentic, test_tampered, [salt]),
                #"test_exposed": self._generate_examples("test", test_authentic, test_tampered, [exposed]),
                "test_compressed": self._generate_examples("test", test_authentic, test_tampered, [compressed]),
                #"test_speckle": self._generate_examples("test", test_authentic, test_tampered, [speckle]),
                #"test_gaussian": self._generate_examples("test", test_authentic, test_tampered, [gaussian]),
                #"test_poisson": self._generate_examples("test", test_authentic, test_tampered, [poisson]),
                }

    def _generate_examples(self,name:str,authentic_files: list, tampered_files: list, modifications: list):
        """Generator of examples for each split.
          :param name: name of the set used for logginf informations
          :param tampered_files: tampered files to include in this set
          :param modifications: set of modifications that should be applied to the elements in this set
        """

        # let's make sure the set is balance
        assert (len(authentic_files) == len(tampered_files))

        # import authentic images
        counter_authentic = 0
        for authentic_img in authentic_files:
            # generate the data of the sample
            sample = self._process_image(authentic_img, False, modifications)

            counter_authentic += 1
            yield str(authentic_img), sample

        # import tampered images
        counter_tampered = 0
        for tampered_img in tampered_files:

            # generate the data of the sample
            sample = self._process_image(tampered_img, True, modifications)

            counter_tampered +=1
            yield str(tampered_img), sample

        print("Dataset: {} contains {} pristine and {} tampered images".format(name,counter_authentic,counter_tampered))

    def _keep_valid(self, files_paths: list):
        """
        Given a list of files return a list only of those that are valid
        :param files_paths:
        :return:
        """
        list = []

        set_of_shapes = {}

        for file_path in files_paths:

            image = Image.open(file_path).convert('RGB')
            image = np.asarray(image)

            if str(image.shape) not in set_of_shapes:
                set_of_shapes[str(image.shape)] = 0

            set_of_shapes[str(image.shape)] += 1

            # check that the image is of the right dimension (or directly transformable to)
            if image.shape[0] not in self.supported_shape or image.shape[1] not in self.supported_shape:
                continue

            list.append(file_path)

        # print data about the dataset
        if self.verbose:
            for key in set_of_shapes:
                print("Found {} images of size {}".format(set_of_shapes[key], key))

            print("Valid images:{}".format(len(list)))

        return list

    def _process_image(self, path, is_tampered, modifications=[]):
        """
        Read the data of an image and the corresponding mask, it the image is authentic, the mask should be black
        :param path: path of the image
        :param is_tampered: bool denoting if an image is tampered or not
        :param modifications: set of modification to apply at the given image before generating the various maps
        :return: n[ array w x h x 3 containing the data of the image and a w x h x 1 np array containing the mask
        """

        target_shape = (256, 384)

        image = Image.open(path).convert('RGB')
        image = np.asarray(image)

        # apply modifications to the image
        for modification in modifications:
            image = modification.apply(image)

        # if the image is flipped rotate it and all its noise maps
        flipped = False
        if image.shape[0] == target_shape[1]:
            image = np.rot90(image, 3)
            flipped = True

        # extract noiseprint map
        noiseprint = self.NoiseprintGenerator.extract(image)

        # extract SRM map
        srm = self.SRMGenerator.extract(image)

        assert (image.shape[0] == self.supported_shape[0])
        assert (image.shape[1] == self.supported_shape[1])
        assert (image.shape[2] == 3)

        assert (image.shape == srm.shape)
        assert (image.shape[0] == noiseprint.shape[0])
        assert (image.shape[1] == noiseprint.shape[1])

        return {"image": image, "noiseprint": noiseprint, "SRM": srm, "flipped": flipped, "tampered": is_tampered}

    def _fix_files_name(self, path_tampered: Path, path_sheet: Path):
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

    def _check_mask(self, tampered_image: Path) -> bool:
        """
        Check if a mask is available for a given tampered image
        :param tampered_image: tampered image whose mask we hae to find
        :return: Boolean
        """
        mask_path = self.extracted_path_gt / (splitext(basename(tampered_image))[0] + "_gt.png")

        if not mask_path.is_file():
            return False

        return True
