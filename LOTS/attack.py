import os
import time
import warnings
import csv

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Ulitities.Plots import plot_graph
from noiseprint2 import gen_noiseprint, NoiseprintEngine, normalize_noiseprint
from noiseprint2.noiseprint_blind import noiseprint_blind_post, genMappFloat, genMappUint8
import tensorflow as tf

from noiseprint2.utility.visualization import image_gt_noiseprint_heatmap_visualization


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred))) / 2


def evaluate_heatmaps(attacked_heatmap: np.array, ground_truth: np.array) -> bool:
    """
    Function to evaluate an attacked heatmap and test if the attack has been succesfull
    :param original_heatmap: heatmap of the original image without adversarial attack
    :param ground_truth: ground truth map of the tamered region
    :return: boolean value
    """

    return attacked_heatmap[ground_truth != 0].mean() < attacked_heatmap[(1 - ground_truth) != 0].mean() and np.max(
        attacked_heatmap) < 50


def attack_noiseprint_model(image, ground_truth, target, QF, steps, debug_folder=None) -> np.array:
    """
    Function to try to break the noiseprint identification mechanism on an imahe
    :param image: image that is the subject of the attack
    :param ground_truth: mask of the forged area
    :param target: target noiseprint
    :param QF: quality factor to use
    :param steps: maximum number of steps to perform
    :param debug: activate the debug mode
    :param debug_folder: folder in which to save logs
    :return:
    """

    # noiseprint when no adversarial attack is performed on the imgage
    original_noiseprint = gen_noiseprint(image, QF)
    attacked_noiseprint = original_noiseprint

    # heatmap when no adversarial attack is performed on the image
    mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(original_noiseprint, image)
    original_heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)
    attacked_heatmap = original_heatmap

    # attacked image
    original_image = image
    attacked_image = image

    # load the model of the noiseprint directly in order to gain access to the gradient
    engine = NoiseprintEngine()
    engine.load_quality(QF)

    # variable to use to store the gradient of the image
    target = tf.cast(tf.convert_to_tensor(target), tf.float32)

    # create csv file to store logs
    csv_file = None
    if debug_folder:
        csv_file = csv.writer(open(os.path.join(debug_folder, 'log.csv'), 'w'))
        csv_file.writerow(['Heatmap min', 'Heatmap mean', 'Heatmap max'])

    # create list to hold changes in the heatmap
    heatmap_mins = []
    heatmap_mins_graph_path = os.path.join(debug_folder, "Plots", "min heatmap value")

    heatmap_means = []
    heatmap_means_graph_path = os.path.join(debug_folder, "Plots", "mean heatmap value")

    heatmap_maxs = []
    heatmap_maxs_graph_path = os.path.join(debug_folder, "Plots", "max heatmap value")

    loss_values = []
    loss_graph_path = os.path.join(debug_folder, "Plots", "loss")
    temperature = 0.005

    for iteration_counter in tqdm(range(steps)):

        # save the image for debug purposes
        if debug_folder:
            img_path = os.path.join(debug_folder, "{}.png".format(str(iteration_counter + 1)))
            image_gt_noiseprint_heatmap_visualization(attacked_image, ground_truth, np.squeeze(attacked_noiseprint),
                                                      attacked_heatmap, img_path)

        with tf.GradientTape() as tape:
            tensor_attacked_image = tf.convert_to_tensor(attacked_image[np.newaxis, :, :, np.newaxis])
            tape.watch(tensor_attacked_image)

            # perform feed foward pass
            attacked_noiseprint = engine._model(tensor_attacked_image)

            # compute the loss with respect to the target representation
            loss = euclidean_distance_loss(target, attacked_noiseprint)

        # compute the gradient
        gradient = tape.gradient(loss, tensor_attacked_image).numpy()

        # generate the attacked heatmap
        mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(np.squeeze(attacked_noiseprint.numpy()),
                                                                            attacked_image)
        attacked_heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

        #save logs and print graphs
        if csv_file:
            csv_file.writerow([attacked_heatmap.min(), attacked_heatmap.mean(), attacked_heatmap.max()])

        heatmap_mins.append(attacked_heatmap.min())
        plot_graph(heatmap_mins, "Heatmap minimum value", heatmap_mins_graph_path)

        heatmap_means.append(attacked_heatmap.mean())
        plot_graph(heatmap_means, "Heatmap mean value", heatmap_means_graph_path)

        heatmap_maxs.append(attacked_heatmap.max())
        plot_graph(heatmap_maxs, "Heatmap max value", heatmap_maxs_graph_path)

        loss_values.append(loss.numpy())
        plot_graph(loss_values, "Loss value", loss_graph_path)

        # check if the attack has been successful
        if evaluate_heatmaps(attacked_heatmap, ground_truth):
            print("The attack was succesfull")

            # if the initial iteration already present no traces in the heatmap print a warning to make it known
            # Does the evaluate_heatmaps function need to be tuned maybe?
            if iteration_counter == 0:
                warnings.warn("No attack is needed to break on this image")

            # The image has been attacked succesfully, return usefull data
            return attacked_image, original_noiseprint, np.squeeze(attacked_noiseprint), \
                   original_heatmap, attacked_heatmap

        # the attack has been unsuccessful, use the gradient to modify the image
        # perform element-wise division
        gradient = gradient / ((np.abs(gradient)).max())

        # apply perturbation on the attacked image
        attacked_image -= np.squeeze(gradient * temperature)

    return False
