from Detectors.Exif.lib.utils.util import process_im
from Utilities.Image.Picture import Picture


def prepare_image(image:Picture):
    return process_im(image)
