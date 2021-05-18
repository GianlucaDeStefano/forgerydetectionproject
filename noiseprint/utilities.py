import numpy as np
import os
from noiseprint.utility.utilityRead import resizeMapWithPadding


def normalize_noiseprint(noiseprint, margin=34):
    """
    Normalize the noiseprint between 0 and 1, in respect to the central area
    :param noiseprint: noiseprint data, 2-D numpy array
    :param margin: margin size defining the central area, default to the overlap size 34
    :return: the normalized noiseprint data, 2-D numpy array with the same size of the input noiseprint data
    """
    v_min = np.min(noiseprint[margin:-margin, margin:-margin])
    v_max = np.max(noiseprint[margin:-margin, margin:-margin])
    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)



def genMappUint8(mapp, valid, range0, range1, imgsize, vmax=None, vmin=None):
    mapp_s = np.copy(mapp)
    mapp_s[valid == 0] = np.min(mapp_s[valid > 0])

    if vmax is None:
        vmax = np.nanmax(mapp_s)
    if vmin is None:
        vmin = np.nanmin(mapp_s)

    mapUint8 = (255 * (mapp_s.clip(vmin, vmax) - vmin) / (vmax - vmin)).clip(0, 255).astype(np.uint8)
    #mapUint8 = 255 - resizeMapWithPadding(mapUint8, range0, range1, imgsize)

    return mapUint8