from Datasets.Casia2.utility import get_authentic_images as get_authentic_images_casia
from Datasets.Casia2.utility import get_forgered_images as get_forgered_images_casia
from Datasets.Casia2.utility import get_mask_of_image as get_mask_of_image_casia2
from Datasets.Columbia.utility import get_authentic_images as get_authentic_images_columbia

from Datasets.Columbia.utility import get_forgered_images as get_forgered_images_columbia

supported_datasets = {
    "columbia": {"authetic": get_authentic_images_columbia,
                 "forged": get_forgered_images_columbia
                 },
    "casia2": {"authetic": get_authentic_images_casia,
               "forged": get_forgered_images_casia,
               "mask": get_mask_of_image_casia2
               }
}
