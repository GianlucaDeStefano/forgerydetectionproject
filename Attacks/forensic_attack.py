import os
import csv
from math import ceil

import PIL
import cv2
import numpy as np
from PIL.Image import Image
from tqdm import tqdm

from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import mask_2_binary
from LOTS.patches import get_authentic_patches
from Ulitities.Images import noise_to_3c
from Ulitities.Plots import plot_graph
from noiseprint2 import NoiseprintEngine
from Detectors.Noiseprint.Noiseprint.noiseprint_blind import noiseprint_blind_post, genMappFloat
import tensorflow as tf

from Detectors.Noiseprint.Noiseprint.utility.utilityRead import imread2f
from Detectors.Noiseprint.Noiseprint.utility import noiseprint_visualization, image_noiseprint_noise_heatmap_visualization, \
    plain_noiseprint_visualization


def get_gradient(image: np.array, image_target_representation: np.array, noiseprint_engine):
    """
    Given an image, a taregt patch representation and a noiseprint model, this image compute the gradient to
    move the image towards the target representation,
    :param image: np.array, image on which to perform the attack
    :param target_patch_representation: target re
    presentation
    :param noiseprint_engine: noiseprint model target of the attack
    :return: tensor containing the gradient,loss
    """

    with tf.GradientTape() as tape:
        tensor_image = tf.convert_to_tensor(image[np.newaxis, :, :, np.newaxis])
        tape.watch(tensor_image)

        # perform feed foward pass
        noiseprint = tf.squeeze(noiseprint_engine._model(tensor_image))

        # compute the loss with respect to the target representation
        loss = tf.nn.l2_loss(image_target_representation - noiseprint)

        # retrieve the gradient of the patch
        image_gradient = np.squeeze(tape.gradient(loss, tensor_image).numpy())

        loss = loss.numpy()

    # scale the final gradient using the computed infinity norm
    image_gradient = image_gradient / np.max(np.abs(image_gradient))

    return image_gradient, loss


def attack_noiseprint_model(image_path, ground_truth_path, QF, steps, debug_folder=None) -> np.array:
    """
    Function to try to break the noiseprint identification mechanism on an imahe
    :param image_path: image that is the subject of the attack
    :param ground_truth_path: mask of the forged area
    :param QF: quality factor to use
    :param steps: maximum number of steps to perform
    :param debug_folder: folder in which to save logs
    :return:
    """
    # load the image to attttack as a 3 channel image
    original_image3c, mode = imread2f(image_path, channel=3)

    # load the image to attack as a 2d array
    original_image, mode = imread2f(image_path)

    # attacked image
    attacked_image = np.copy(original_image)

    # load the groundtruth as a 2d array
    ground_truth = PIL.Image.open(ground_truth_path)
    ground_truth.load()
    ground_truth = mask_2_binary(ground_truth)

    # load the model of the noiseprint directly in order to gain access to the gradient
    engine = NoiseprintEngine()
    engine.load_quality(QF)

    # noiseprint when no adversarial attack is performed on the imgage
    original_noiseprint = engine.predict(original_image)
    attacked_noiseprint = original_noiseprint

    # attacked image
    attacked_heatmap = None

    # create csv file to store logs
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

    variance_values = []
    variance_graph_path = os.path.join(debug_folder, "Plots", "variance")

    psnr_values = []
    psnr_graph_path = os.path.join(debug_folder, "Plots", "psnr")

    # get patches of image
    patch_size = (8, 8)

    authentic_patches = get_authentic_patches(original_noiseprint, ground_truth, patch_size, True)

    # representation of the ideal patch
    target_patch = np.zeros(patch_size)

    # cumulative gradient
    cumulative_gradient = np.zeros(original_image.shape)

    # generate authentic target representation
    print("Generating target representation...")
    for x_index, y_index, patch in tqdm(authentic_patches):
        target_patch += patch / len(authentic_patches)

    noiseprint_visualization(target_patch, os.path.join(debug_folder, "target.png"))
    plain_noiseprint_visualization(target_patch,os.path.join(debug_folder, "plain-target.png"))
    # divide the entire image to attack into patches
    repeat_factors = (ceil(original_image.shape[0] / target_patch.shape[0]), ceil(original_image.shape[1] / target_patch.shape[1]))
    image_target_representation = np.tile(target_patch, repeat_factors)
    image_target_representation = image_target_representation[:original_image.shape[0], :original_image.shape[1]]

    plain_noiseprint_visualization(image_target_representation, os.path.join(debug_folder, "image-target.png"))
    alpha = 5
    for iteration_counter in tqdm(range(steps)):

        # save the image for debug purposes
        if debug_folder:
            # generate the noiseprint to visualize on the entire image
            attacked_noiseprint = engine.predict(attacked_image)

            if debug_folder:
                # generate the attacked heatmap
                mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(attacked_noiseprint, attacked_image)
                attacked_heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

                heatmap_mins.append(attacked_heatmap.min())
                plot_graph(heatmap_mins, "Heatmap minimum value", heatmap_mins_graph_path)

                heatmap_means.append(attacked_heatmap.mean())
                plot_graph(heatmap_means, "Heatmap mean value", heatmap_means_graph_path)

                heatmap_maxs.append(attacked_heatmap.max())
                plot_graph(heatmap_maxs, "Heatmap max value", heatmap_maxs_graph_path)

                variance_values.append(np.var(attacked_noiseprint))
                plot_graph(variance_values, "Variance", variance_graph_path)

                psnr_values.append(cv2.PSNR(original_image, attacked_image))
                plot_graph(psnr_values, "PSNR", psnr_graph_path)

                plot_graph(loss_values, "Loss value", loss_graph_path,initial_value=1)

                # save image
                img_path = os.path.join(debug_folder, "{}.png".format(str(iteration_counter)))
                adversarial_noise = noise_to_3c(cumulative_gradient)
                attacked_image3c = (original_image3c + adversarial_noise)
                image_noiseprint_noise_heatmap_visualization(attacked_image3c, np.squeeze(attacked_noiseprint),
                                                             cumulative_gradient, attacked_heatmap, img_path)

        # compute the gradient
        image_gradient, loss = get_gradient(attacked_image, image_target_representation, engine)

        # append the loss values to the list
        loss_values.append(loss)

        # scale the gradient in the desided way
        gradient = alpha * image_gradient / 255

        # add gradient to the cumulative gradient
        cumulative_gradient += gradient

        # apply the gradient to the image
        attacked_image -= gradient

        # clip image values in the valid range
        attacked_image = np.clip(attacked_image, 0, 1)


    return False
