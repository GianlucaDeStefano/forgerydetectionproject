from Datasets.Columbia.ColumbiaDataset import ColumbiaDataset
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Datasets.Dataset import ImageNotFoundException
from Datasets.RIT.RitDataset import RitDataset

supported_datasets = dict(columbia=ColumbiaDataset, rit=RitDataset, columbiaUncompressed=ColumbiaUncompressedDataset
                          , dso=DsoDatasetDataset)


def find_dataset_of_image(image_name):
    """
    Look for an image with a specific name inside each dataset, return the first dataset found containing an image with
    the  corresponding name
    :return:
    """

    for key, candidate_dataset in supported_datasets.items():
        try:
            if candidate_dataset().get_image(image_name):
                return candidate_dataset()
        except ImageNotFoundException as e:
            continue

    return False
