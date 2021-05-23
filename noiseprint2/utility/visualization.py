import numpy as np
from matplotlib import pyplot as plt, gridspec
from sklearn import metrics

from noiseprint2 import gen_noiseprint, normalize_noiseprint
from noiseprint2.noiseprint_blind import genMappFloat, noiseprint_blind
from noiseprint2.utility.utilityRead import imread2f, computeMCC

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter

erodeKernSize = 15
dilateKernSize = 11


def full_image_visualization(image_path,gt_path, QF = None, output_path: str = None):
    # load the image
    img, mode = imread2f(image_path, channel = 3)

    gt = imread2f(gt_path, channel = 1)[0]>0.5
    print('size : ', img.shape)
    assert(img.shape[0]==gt.shape[0])
    assert(img.shape[1]==gt.shape[1])

    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    gtV = np.logical_or(gt0, gt1)

    plt.figure(figsize=(4 * 5, 2 * 5))
    grid = gridspec.GridSpec(2, 4, wspace=0.2, hspace=0.2, )

    plt.subplot(grid[0, 0])
    plt.imshow(img, clim=[0, 1])
    plt.title('Input image')

    mapp, valid, range0, range1, imgsize, other,noiseprint = noiseprint_blind(image_path,QF)

    heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

    plt.subplot(grid[0, 1])
    plt.imshow(normalize_noiseprint(noiseprint), clim=[0, 1], cmap='gray')
    plt.title('Noiseprint')

    plt.subplot(grid[0, 2])
    plt.imshow(gt, clim=[0, 1], cmap='gray')
    plt.title('Ground truth')

    plt.subplot(grid[0, 3])
    plt.imshow(heatmap, clim=[np.nanmin(heatmap), np.nanmax(heatmap)], cmap='jet')
    plt.title('Heatmap')

    mcc, ths = computeMCC(heatmap, gt0, gt1)
    plt.subplot(grid[1, :2])
    plt.plot(ths, mcc)
    plt.grid()
    plt.xlabel('threshold')
    plt.ylabel('|MCC|')
    plt.legend(['max |MCC|=%5.3f' % np.max(mcc)])
    plt.title('Matthews Correlation Coefficient')

    ap1 = metrics.average_precision_score(gt1[gtV], +heatmap[gtV])
    ap2 = metrics.average_precision_score(gt1[gtV], -heatmap[gtV])
    smapp = heatmap if ap1 >= ap2 else -heatmap
    ap = max(ap1, ap2)
    prec, racall, _ = metrics.precision_recall_curve(gt1[gtV], smapp[gtV])

    plt.subplot(grid[1, 2])
    plt.plot(racall, prec)
    plt.grid();
    plt.xlabel('recall ');
    plt.ylabel('precision')
    plt.legend(['AP=%5.3f' % ap])
    plt.title('precision-recall curve')

    plt.show()

    if(output_path):
        plt.savefig(output_path)

    return plt
