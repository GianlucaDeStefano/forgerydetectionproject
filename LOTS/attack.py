import os
import time
import warnings

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from noiseprint2 import gen_noiseprint, NoiseprintEngine, normalize_noiseprint
from noiseprint2.noiseprint_blind import noiseprint_blind_post, genMappFloat
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
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true)))

def evaluate_heatmaps(attacked_heatmap: np.array, ground_truth:np.array) -> bool:
    """
    Function to evaluate an attacked heatmap and test if the attack has been succesfull
    :param original_heatmap: heatmap of the original image without adversarial attack
    :param ground_truth: ground truth map of the tamered region
    :return: boolean value
    """

    #very basic evaluation procedure to change in the future with a better one
    tampered_heatmap = attacked_heatmap * ground_truth
    authentic_heatmap = attacked_heatmap *(1-ground_truth)


    return tampered_heatmap[ground_truth!=0].mean() < authentic_heatmap[(1-ground_truth)!=0].mean()


def attack_noiseprint_model(image,ground_truth, target, QF, steps,debug_folder=None) -> np.array:
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
    mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(original_noiseprint,image)
    original_heatmap = genMappFloat(mapp, valid, range0,range1, imgsize)
    attacked_heatmap = original_heatmap

    # attacked image
    attacked_image = image

    #load the model of the noiseprint directly in order to gain access to the gradient
    engine = NoiseprintEngine()
    engine.load_quality(QF)

    #variable to use to store the gradient of the image
    target = tf.cast(tf.convert_to_tensor(target), tf.float32)

    for iteration_counter in tqdm(range(steps)):

        #save the image for debug purposes
        if debug_folder:
            img_path = os.path.join(debug_folder, "{}.png".format(str(iteration_counter + 1)))
            image_gt_noiseprint_heatmap_visualization(attacked_image,ground_truth,np.squeeze(attacked_noiseprint),attacked_heatmap,img_path)

        with tf.GradientTape() as tape:
            tensor_attacked_image = tf.convert_to_tensor(attacked_image[np.newaxis, :, :, np.newaxis])
            tape.watch(tensor_attacked_image)

            #perform feed foward pass
            attacked_noiseprint = engine._model(tensor_attacked_image)

            # compute the loss with respect to the target representation
            #loss = tf.multiply(tf.pow(tf.subtract(target, attacked_noiseprint), 2), 1 / 2)
            loss  = euclidean_distance_loss(target,attacked_noiseprint)

        #compute the gradient
        gradient = tape.gradient(loss, tensor_attacked_image).numpy()

        #generate the attacked heatmap
        mapp, valid, range0, range1, imgsize, other = noiseprint_blind_post(np.squeeze(attacked_noiseprint.numpy()), attacked_image)
        attacked_heatmap = genMappFloat(mapp, valid, range0, range1, imgsize)

        # check if the attack has been successful
        if evaluate_heatmaps(attacked_heatmap,ground_truth):
            print("The attack was succesfull")

            #if the initial iteration already present no traces in the heatmap print a warning to make it known
            #Does the evaluate_heatmaps function need to be tuned maybe?
            if iteration_counter == 0:
                warnings.warn("No attack is needed to break on this image")

            # The image has been attacked succesfully, return usefull data
            return attacked_image, original_noiseprint, np.squeeze(attacked_noiseprint), \
                   original_heatmap, attacked_heatmap

        #the attack has been unsuccessful, use the gradient to modify the image
        #perform element-wise division
        gradient = gradient/((np.abs(gradient)).max())

        #apply perturbation on the attacked image
        attacked_image -= np.squeeze(gradient*0.01)

    return False