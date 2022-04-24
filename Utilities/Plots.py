import os

from matplotlib import pyplot as plt

def plot_graph(data, label_y, label_x="", path=None, min_range_value=1, initial_value=1):
    """
    Generate and save a cartesian graph displaying the given list of datapoints
    @param data: list of datapoints to display
    @param label: label to print on the x axis
    @param path: path to save the imae
    @param display: should the image be opened when created?
    @param min_range_value: minumum number of values to diaplay as index on the x axis
    @param initial_value: base value on the x axis
    :return:
    """
    plt.close()
    plt.plot(data)

    plt.ylabel(label_y)
    plt.xlabel(label_x)

    plt.xticks(range(min_range_value, len(data), max(initial_value, int(len(data) / 10))))
    if path:
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        plt.savefig(path)

    plt.close()