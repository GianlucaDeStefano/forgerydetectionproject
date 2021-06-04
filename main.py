import numpy as np
from tqdm import tqdm

from Datasets.RIT.RitDataset import RitDataset
from LOTS.patches import divide_in_patches, scale_patch
from noiseprint2.utility.utilityRead import imread2f
import matplotlib.pyplot as plt

original_image, mode = imread2f("./Datasets/RIT/Data/Canon_60D/tampered-realistic/DPP0520.TIF")
patch_shape = (16, 16)

patches = divide_in_patches(original_image, patch_shape, False)

reconstructed_image = np.zeros(original_image.shape)

for x_index, y_index,patch in tqdm(patches):
    reconstructed_image += scale_patch(patch,reconstructed_image.shape, x_index, y_index)


fig, ax = plt.subplots()
ax.imshow(reconstructed_image)

plt.show()
