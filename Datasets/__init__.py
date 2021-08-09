from pathlib import Path

from Datasets.Columbia.ColumbiaDataset import ColumbiaDataset
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Datasets.Dataset import ImageNotFoundException, mask_2_binary
from Datasets.RIT.RitDataset import RitDataset
from Detectors.Noiseprint.utility.utilityRead import imread2f
from Ulitities.Image.Picture import Picture

supported_datasets = dict(columbia=ColumbiaDataset, rit=RitDataset, columbiaUncompressed=ColumbiaUncompressedDataset
                          , dso=DsoDatasetDataset)


def find_dataset_of_image(datasets_root, image_name):
    """
    Look for an image with a specific name inside each dataset, return the first dataset found containing an image with
    the  corresponding name
    :return:
    """

    for key, candidate_dataset in supported_datasets.items():
        try:
            if candidate_dataset(datasets_root).get_image(image_name):
                return candidate_dataset
        except ImageNotFoundException as e:
            continue

    return False


def get_image_and_mask(dataset_root,image_reference, mask_path=None):
    """
    Given a reference to an image (it's name or path), return its path
    :param image_reference: name of an imge belonging to one of the datasets, or a path orf an external image
    :return: path of the image
    """

    # image reference and mask_path are both provided they have to be direct paths
    if mask_path is not None and Path(mask_path).exists():

        if image_reference is None or not Path(image_reference).exists():
            raise Exception("Provided a valid mask path but an invalid image path")

        return Picture(image_reference), Picture(mask_2_binary(imread2f(mask_path)), mask_path)

    # select the first dataset having an image with the corresponding name
    dataset = find_dataset_of_image(dataset_root, image_reference)
    if not dataset:
        raise Exception("Impossible to find the dataset this image belongs to")

    dataset = dataset(dataset_root)

    image_path = dataset.get_image(image_reference)
    mask, mask_path = dataset.get_mask_of_image(image_reference)

    # load the image as a 3 dimensional numpy array
    image = Picture(image_path)

    # load mask as a picture
    mask = Picture(mask, mask_path)

    return image, mask
