import numpy as np

from Ulitities.Image.Picture import Picture

noise = np.ones((10, 10))

p = Picture(noise)

p_3 = Picture(p.three_channel)

print(p_3.one_channel[0,0])
