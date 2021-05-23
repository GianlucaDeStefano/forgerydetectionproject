import os
import time
import warnings

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from noiseprint2 import gen_noiseprint, NoiseprintEngine, normalize_noiseprint
from noiseprint2.noiseprint_blind import noiseprint_blind_post, genMappFloat
import tensorflow as tf

def euclidean_loss(target,current):
    return ((target - current)**2)/2



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

    return tampered_heatmap[ground_truth!=0].mean() < authentic_heatmap[(1-ground_truth)!=0].mean() and tampered_heatmap[ground_truth!=0].mean() <0.1


def attack_noiseprint_model(image,ground_truth, target, QF, steps, debug:bool = False,debug_folder="./Data/Debug/") -> np.array:

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

    start_time = time.time()
    debug_folder = os.path.join(debug_folder, str(start_time))
    if debug:
        os.makedirs(debug_folder)

    for iteration_counter in tqdm(range(steps)):

        #save the image for debug purposes
        if debug:
            fig, axs = plt.subplots(1,3)
            axs[0].imshow(attacked_image)
            axs[0].set_title('Attacked image')
            axs[1].imshow(normalize_noiseprint(np.squeeze(attacked_noiseprint)))
            axs[1].set_title('Attacked noiseprint')
            axs[2].imshow(attacked_heatmap)
            axs[2].set_title('Attacked heatmap')

            plt.savefig(os.path.join(debug_folder,"{}.png".format(iteration_counter+1)))
            plt.close(fig)

        with tf.GradientTape() as tape:
            tensor_attacked_image = tf.convert_to_tensor(attacked_image[np.newaxis, :, :, np.newaxis])
            tape.watch(tensor_attacked_image)

            #perform feed foward pass
            attacked_noiseprint = engine._model(tensor_attacked_image)

            # compute the loss with respect to the target representation
            loss = tf.multiply(tf.pow(tf.subtract(target, attacked_noiseprint), 2), 1 / 2)

        #compute the gradient
        gradient = tape.gradient(loss, tensor_attacked_image)

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

            # The image has been attacked succesfully
            return attacked_image, original_noiseprint, np.squeeze(attacked_noiseprint), \
                   original_heatmap, attacked_heatmap

        #the attack has been unsuccessful, use the gradient to modify the image
        #perform element-wise division
        gradient = gradient/((np.abs(gradient)).max())

        #apply perturbation on the attacked image
        attacked_image -= np.squeeze(gradient*0.05)

    return False