import numpy as np
from matplotlib import pyplot as plt

def visuallize_matrix_values(matrix, path):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            c = matrix[i, j]
            ax.text(j, i, "{:.2f}".format(c), va='center', ha='center', size=7)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(path, bbox_inches='tight', dpi=600)


