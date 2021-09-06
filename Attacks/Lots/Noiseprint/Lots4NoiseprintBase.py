import os.path
import os.path
from abc import ABC, abstractmethod
import cv2
import numpy as np
import tensorflow as tf

# setup tensorflow session conf
from Detectors.Noiseprint.utility.utility import jpeg_quality_of_file
from Ulitities.Visualizers.NoiseprintVisualizer import NoiseprintVisualizer

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from Attacks.Lots.BaseLotsAttack import BaseLotsAttack
from Detectors.Noiseprint.noiseprintEngine import NoiseprintEngine
from Ulitities.Image.Patch import Patch
from Ulitities.Image.Picture import Picture


class MissingTargetRepresentation(Exception):
    def __init__(self, image_name):
        super().__init__("No image found with name: {}".format(image_name))


def lots_loss(y_pred, y_true):
    return tf.reduce_sum(tf.abs(y_true - y_pred))**2 / 2



def normalize_gradient(gradient, margin=17):
    """
    Normalize the gradient cutting away the values on the borders
    :param margin: margin to use along the bordes
    :param gradient: gradient to normalize
    :return: normalized gradient
    """

    # set to 0 part of the gradient too near to the border
    if margin > 0:
        gradient[0:margin, :] = 0
        gradient[-margin:, :] = 0
        gradient[:, 0:margin] = 0
        gradient[:, -margin:] = 0

    # scale the final gradient using the computed infinity norm
    gradient = gradient / np.max(np.abs(gradient))
    return gradient


class Lots4NoiseprintBase(BaseLotsAttack, ABC):

    name = "LOTS4NoiseprintBase"

    def __init__(self, objective_image: Picture, objective_mask: Picture,
                 target_representation_image: Picture = None,
                 target_representation_mask: Picture = None, qf: int = None,
                 patch_size: tuple = (8, 8), steps=50,
                 debug_root="./Data/Debug/", alpha=5, plot_interval=3, verbose=True):
        """
        Base class to implement various attacks
        :param objective_image: image to attack
        :param objective_mask: binary mask of the image to attack, 0 = authentic, 1 = forged
        :param qf: quality factor to use
        :param patch_size: size of the patch ot use to generate the target representation
        :param steps: total number of steps of the attack
        :param debug_root: root folder in which save debug data generated by the attack
        """
        # check the passe patch is not too wide for be handled by noiseprint
        assert (patch_size[0] * patch_size[1] < NoiseprintEngine.large_limit)

        super().__init__(objective_image, objective_mask, target_representation_image, target_representation_mask,
                         patch_size, steps, debug_root, alpha,
                         plot_interval, verbose, None)

        if not qf or qf < 51 or qf > 101:
            try:
                print(objective_image.path)
                qf = jpeg_quality_of_file(objective_image.path)
            except:
                qf = 101

        # save the parameters of noiseprint
        self.qf = qf

        self._engine = NoiseprintEngine()
        self._engine.load_quality(qf)

        self.loss_steps = []
        self.psnr_steps = []
        self.noiseprint_variance_steps = []

        self.min_loss = float("inf")
        self.visualizer = NoiseprintVisualizer(self.qf)

    def _on_after_attack_step(self):

        # compute PSNR between intial 1 channel image and the attacked one
        psnr = cv2.PSNR(self.original_objective_image.one_channel(), self.attacked_image.one_channel())
        self.psnr_steps.append(psnr)

        # compute variance on the noiseprint map
        image = self.attacked_image.one_channel().to_float()
        noiseprint = self._engine.predict(image)
        self.noiseprint_variance_steps.append(noiseprint.var())

        super()._on_after_attack_step()

        self.visualizer.plot_graph(self.loss_steps, "Loss", "Attack iteration", os.path.join(self.debug_folder, "loss"))
        self.visualizer.plot_graph(self.psnr_steps, "PSNR", "Attack iteration", os.path.join(self.debug_folder, "psnr"))
        self.visualizer.plot_graph(self.noiseprint_variance_steps, "Attack iteration", "Variance",
                                   os.path.join(self.debug_folder, "variance"))

        if self.loss_steps[-1] < self.min_loss:
            self.min_loss = self.loss_steps[-1]

            self.best_noise = self.original_objective_image - self.attacked_image

            # Log
            self.write_to_logs(
                "New optimal noise found, saving it to :{}".format(os.path.join(self.debug_folder, 'best-noise.npy')))

            # save the best adversarial noise
            np.save(os.path.join(self.debug_folder, 'best-noise.npy'), self.best_noise)

    def plot_step(self, image, path):
        """
        Print for debug purposes the state of the attack
        :return:
        """

        if not self.debug_folder or not self.visualizer:
            return

        psnr = 999
        if len(self.psnr_steps) > 0:
            psnr = self.psnr_steps[-1]

        note = "Step:{}, PSNR:{:.2f}".format(self.attack_iteration,psnr)
        self.visualizer.prediction_pipeline(image.to_float(),
                                            path,original_picture= self.original_objective_image.one_channel().to_float()
                                            ,note=note,omask=self.objective_image_mask,debug=False,
                                            adversarial_noise=self.original_objective_image.to_float() - self.attacked_image.to_float())


    def _get_gradient_of_patch(self, image_patch: Patch, target):
        """
        Compute gradient of the patch
        :param image_patch:
        :param target:
        :return:
        """

        assert (image_patch.shape == target.shape)

        # be sure that the given patch and target are of the same shape
        with tf.GradientTape() as tape:
            tensor_patch = tf.convert_to_tensor(image_patch[np.newaxis, :, :, np.newaxis])
            tape.watch(tensor_patch)

            # perform feed foward pass
            patch_noiseprint = tf.squeeze(self._engine._model(tensor_patch))

            # compute the loss with respect to the target representation
            loss = self.loss(patch_noiseprint,target)

            # retrieve the gradient of the patch
            patch_gradient = np.squeeze(tape.gradient(loss, tensor_patch).numpy())

            # check that the retrieved gradient has the correct shape
            assert (patch_gradient.shape == image_patch.shape)

            return patch_gradient, loss

    @abstractmethod
    def _get_gradient_of_image(self, image: Picture, target: Picture):
        """
        Compute the gradient for the entire image
        :param image: image for which we have to compute the gradient
        :param target: target to use
        :return: numpy array containing the gradient
        """

        raise NotImplemented

    def _attack(self):
        """
        Perform step of the attack executing the following steps:

            1) Divide the entire image into patches
            2) Compute the gradient of each patch with respect to the patch-tirget representation
            3) Recombine all the patch-gradients to obtain a image wide gradient
            4) Apply the image-gradient to the image
            5) Convert then the image to the range of values of integers [0,255] and convert it back to the range
               [0,1]
        :return:
        """

        # compute the attacked image using the original image and the compulative noise to reduce
        # rounding artifacts caused by translating the nosie from one to 3 channels and vie versa multiple times
        image = self.attacked_image_monochannel

        # compute the gradient
        image_gradient, loss = self._get_gradient_of_image(image,self.target_representation)

        # save loss value to plot it
        self.loss_steps.append(loss)

        # normalize the gradient
        image_gradient = normalize_gradient(image_gradient, 0)

        # scale the gradient
        image_gradient = self.alpha * image_gradient

        self.noise += image_gradient

        # add noise
        image -= Picture(image_gradient).three_channels()

        return image

    def _log_step(self) -> str:
        "Generate the logging to write at each step"
        return " {}) Duration: {} Loss:{} BestLoss:{}".format(self.attack_iteration,
                                                              self.end_step_time - self.start_step_time,
                                                              self.loss_steps[-1], min(self.loss_steps))

    @abstractmethod
    def _generate_target_representation(self, image: Picture, mask: Picture):
        raise NotImplemented

    def loss(self,prediction,target):
        return tf.reduce_mean(tf.math.squared_difference(target, prediction))