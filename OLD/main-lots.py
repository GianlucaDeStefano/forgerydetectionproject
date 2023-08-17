from Attacks.Exif.Lots.Lots4Exif_original import Lots4ExifOriginal
from Utilities.Confs.Configs import Configs
from Attacks.Noiseprint.Lots.Lots4Noiseprint_original import Lots4NoiseprintAttackOriginal
from Datasets import get_image_and_mask

image_to_attack = "splicing-70.png"

# Load config file
configs = Configs("config.yaml", "GenericTest")

#Load image
image,mask = get_image_and_mask(configs["global"]["datasets"]["root"], image_to_attack)

# instance the attack
attack = Lots4ExifOriginal(50,5,plot_interval=-1)

attack.setup(image.path,target_image_mask=mask)

last,best = attack.execute()