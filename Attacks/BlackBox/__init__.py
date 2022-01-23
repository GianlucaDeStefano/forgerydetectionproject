from .GaussianNoiseAddition import GaussianNoiseAdditionAttack
from .JpegCompression import JpegCompressionAttack

# list of supported black box attacks
black_box_attacks = {
    "Gaussian Noise Addition": GaussianNoiseAdditionAttack,
    "Jpeg compression": JpegCompressionAttack
}