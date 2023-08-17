import numpy as np

heatmap = np.array([(i)/10 for i in range(11)])

print(heatmap)

percentile = np.quantile(np.array(heatmap), 0.8)

print(percentile)

mask = np.where(heatmap >= percentile, 1, 0)

print(mask)