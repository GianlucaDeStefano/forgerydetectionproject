from Attacks.AdversarialAttacks.GaussianNoiseAddition import GaussianNoiseAdditionAttack
from Attacks.AdversarialAttacks.JpegCompression import JpegCompressionAttack
from Attacks.Lots.Noiseprint.Lots4Noiseprint1.Lots4Noiseprint1 import LotsNoiseprint1
from Attacks.Lots.Noiseprint.Lots4Noiseprint1.Lots4Noiseprint1B import LotsNoiseprint1B
from Attacks.Lots.Noiseprint.Lots4Noiseprint2.Lots4Noiseprint2 import LotsNoiseprint2
from Attacks.Lots.Noiseprint.Lots4Noiseprint3.Lots4Noiseprint3 import LotsNoiseprint3

supported_attacks = {
    "Lots4Noiseprint.1 passing of 32 px": LotsNoiseprint1,
    "Lots4Noiseprint.1.b no padding": LotsNoiseprint1B,
    "Lots4Noiseprint.2 one step gradient": LotsNoiseprint2,
    "Lots4Noiseprint.3 Transfer noisepritn": LotsNoiseprint3,
    "GaussianNoiseAddition": GaussianNoiseAdditionAttack,
    "JpegCompression": JpegCompressionAttack,
}
