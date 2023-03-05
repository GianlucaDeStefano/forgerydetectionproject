import numpy as np

mask = np.array([1, 1, 0])

hetamap = np.array([0, 1, 0])

dr_gt = np.ma.masked_where(mask == 0, hetamap).sum() / mask.sum()

dr_bg = np.ma.masked_where(mask == 1, hetamap).sum() / (mask == 0).sum()

print(dr_gt,dr_bg)
