import os

from matplotlib import pyplot as plt


def plot_graph(data,label,path=None,display=False,min_range_value=1,initial_value=0):

    plt.plot(data)
    plt.ylabel(label)

    plt.xlabel("Attack iteration")
    plt.xticks(range(min_range_value, len(data),max(initial_value,int(len(data)/10))))
    if path:
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        plt.savefig(path)

    if display:
        plt.show()
    plt.close()