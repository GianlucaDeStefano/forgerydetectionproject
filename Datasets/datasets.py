from Datasets.Casia2.utility import get_authentic_images as get_authentic_images_casia
from Datasets.Casia2.utility import get_forgered_images as get_forgered_images_casia
from Datasets.Columbia.utility import get_authentic_images as get_authentic_images_columbia

from Datasets.Columbia.utility import get_forgered_images as get_forgered_images_columbia

supported_datasets = {
    "columbia": {"authetic": get_authentic_images_columbia,
                 "forged": get_forgered_images_columbia
                 },
    "casia2": {"authetic": get_authentic_images_casia,
               "forged": get_forgered_images_casia
               }
}
