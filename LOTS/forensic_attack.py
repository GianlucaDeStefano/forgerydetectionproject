import os
import time
import warnings
import csv

import PIL
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from Datasets.ColumbiaUncompressed.ColumbiaUncompressedDataset import mask_2_binary
from LOTS.patches import get_authentic_patches, divide_in_patches, scale_patch, get_forged_patches
from Ulitities.Images import load_mask, noise_to_3c
from Ulitities.Plots import plot_graph
from noiseprint2 import gen_noiseprint, NoiseprintEngine, normalize_noiseprint
from noiseprint2.noiseprint import normalize_noiseprint_no_margin
from noiseprint2.noiseprint_blind import noiseprint_blind_post, genMappFloat, genMappUint8
import tensorflow as tf

from noiseprint2.utility.utilityRead import imread2f
from noiseprint2.utility.visualization import noiseprint_visualization, image_noiseprint_noise_heatmap_visualization


def get_gradient(image: np.array, mask: np.array, target_representation: np.array, noiseprint_engine):
    """
    Given an image, a taregt patch representation and a noiseprint model, this image compute the gradient to
    move the image towards the target representation,
    :param image: np.array, image on which to perform the attack
    :param target_patch_representation: target re
    presentation
    :param noiseprint_engine: noiseprint model target of the attack
    :return: tensor containing the gradient,loss
    """

    # divide the entire image to attack into patches
    img_patches = divide_in_patches(image, target_representation.shape, False)
    #img_patches = get_forged_patches(image, mask, target_representation.shape, False)
    # create a list to hold the gradients to apply on the image
    gradients = []

    # variable to store the cumulative loss across all patches
    cumulative_loss = 0

    #image wide gradient
    image_gradient = np.zeros(image.shape)

    # analyze the image patch by patch
    pbar = tqdm(img_patches)

    for x_index, y_index, patch in pbar:
        # compute the gradient on the given patch
        with tf.GradientTape() as tape:
            tensor_patch = tf.convert_to_tensor(patch[np.newaxis, :, :, np.newaxis])
            tape.watch(tensor_patch)

            # perform feed foward pass
            patch_noiseprint = tf.squeeze(noiseprint_engine._model(tensor_patch))

            target_patch_representation = np.copy(target_representation)

            if target_patch_representation.shape != patch.shape:
                target_patch_representation = target_patch_representation[:patch.shape[0],:patch.shape[1]]
                continue

            # compute the loss with respect to the target representation
            loss = tf.nn.l2_loss(target_patch_representation-patch_noiseprint)

            # retrieve the gradient of the patch
            patch_gradient = np.squeeze(tape.gradient(loss, tensor_patch).numpy())

            cumulative_loss += loss.numpy()

            # check that the retrieved gradient has the correct shape
            assert (patch_gradient.shape == patch.shape)

            #Add the contribution of this patch to the image wide gradient
            image_gradient += scale_patch(patch_gradient, image_gradient.shape, x_index, y_index)

    # scale the final gradient using the computed infinity norm
    image_gradient = image_gradient / np.max(np.abs(image_gradient))

    return image_gradient, cumulative_loss


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

    # get patches of image
    patch_size = (16, 16)

    authentic_patches = get_authentic_patches(original_image, ground_truth, patch_size, True)

    #representation of the ideal patch
    target_patch = np.zeros(patch_size)

    # generate authentic target representation
    print("Generating target representation...")
    for x_index, y_index, patch in tqdm(authentic_patches):

        patch = np.squeeze(engine._model(patch[np.newaxis, :, :, np.newaxis]))

        target_patch += patch / len(authentic_patches)

    noiseprint_visualization(normalize_noiseprint_no_margin(target_patch), os.path.join(debug_folder, "target"))

    cumulative_gradient = np.zeros(original_image.shape)

    alpha = 5
    for iteration_counter in range(steps):

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

                plot_graph(loss_values, "Loss value", loss_graph_path,initial_value=1)

                # save image
                img_path = os.path.join(debug_folder, "{}.png".format(str(iteration_counter)))
                adversarial_noise = noise_to_3c(cumulative_gradient)
                attacked_image3c = (original_image3c + adversarial_noise)
                image_noiseprint_noise_heatmap_visualization(attacked_image3c, np.squeeze(attacked_noiseprint),
                                                             cumulative_gradient, attacked_heatmap, img_path)

        # compute the gradient
        image_gradient, mean_loss = get_gradient(attacked_image, ground_truth, target_patch, engine)

        #append the loss values to the list
        loss_values.append(mean_loss)

        # scale the gradient in the desided way
        gradient = alpha * image_gradient / 255

        # add gradient to the cumulative gradient
        cumulative_gradient += gradient

        # apply the gradient to the image
        attacked_image -= gradient

        # clip image values in the valid range
        attacked_image = np.clip(attacked_image, 0, 1)


    return False
