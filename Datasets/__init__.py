from Datasets.Columbia.ColumbiaDataset import ColumbiaDataset
from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import ColumbiaUncompressedDataset
from Datasets.DSO.DsoDataset import DsoDatasetDataset
from Datasets.RIT.RitDataset import RitDataset

supported_datasets = dict(columbia=ColumbiaDataset, rit=RitDataset, columbiaUncompressed=ColumbiaUncompressedDataset
                          , dso=DsoDatasetDataset)
