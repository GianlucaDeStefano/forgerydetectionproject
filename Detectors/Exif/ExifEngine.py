import gc
import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
from Detectors.DetectorEngine import DetectorEngine, segment_heatmap, find_optimal_mask
from Detectors.Exif import demo


class ExifEngine(DetectorEngine):
    weights_path = os.path.join(pathlib.Path(__file__).parent, './ckpt/exif_final/exif_final.ckpt')

    def __init__(self, dense=False,use_gpu=0):

        self.patch_per_dim = 30
        self.model = None
        self.dense = dense
        self.use_gpu = use_gpu

        tf.compat.v1.disable_eager_execution()
        super().__init__("ExifEngine")

    def reset_instance(self):
        """
        Destroy & Reload the detector's instance from memory
        @return:
        """
        self.destroy()

        self.model = demo.Demo(ckpt_path=self.weights_path, use_gpu=self.use_gpu, quality=3.0, num_per_dim=self.patch_per_dim)

    def destroy(self):
        """
        @return:
        """
        # check if a session object exists
        if self.model and self.model.solver.sess:

            # close tf section if open
            if not self.model.solver.sess._closed:
                self.model.solver.sess.close()

        del self.model

        gc.collect()

    def reset(self, reset_instance=False, reset_metadata=True):
        """
        Reset Exif to a pristine state
        @return: None
        """

        # call the constructor reset class
        super().reset(reset_instance, reset_metadata)

    def initialize(self, sample_path=None, sample=None, reset_instance=False, reset_metadata=True):
        """
        Initialize the detector to handle a new sample, inferring also the Noiseprint's quality factor to use.

        @param sample_path: str
            Path of the sample to analyze
        @param sample: numpy.array
            Preloaded sample to use (to be useful sample_path has to be None)
        @param reset_instance: Bool
            A flag indicating if this detector's model should be reimported
        @param reset_metadata:Bool
            A flag indicating if the metadata should be reinitialized before loading the new sample
        @param reset_metadata:Bool
            A flag indicating if the metadata should be reinitialized before loading the new sample

        """

        # Generic initialization of the detector engine
        super(ExifEngine, self).initialize(sample_path, sample, reset_instance or self.model is None, reset_metadata)

        if not reset_metadata and sample is not None:
            # In case the metadata have not been completely widped,
            # delete still the entries containing the extracted features to ensure a fair new run
            if "features" in self.metadata: del self.metadata["features"]

        # Make sure the necessary data has been loaded
        assert ("sample_path" in self.metadata.keys() or self.metadata["sample"])

        # Make sure the necessary data has been loaded
        assert (self.metadata["sample_path"] or self.metadata["sample"] is not None)

        # check if a sample instance has been given
        if self.metadata["sample"] is None:
            # no instance given, load it from the path

            # read the necessary metadata
            sample_path = self.metadata["sample_path"]

            # Load the sample
            sample = cv2.imread(sample_path)[:, :, [2, 1, 0]]
            self.metadata["sample"] = sample

    def extract_features(self):
        # check if the features have already been extracted
        if "features" in self.metadata:
            return self.metadata

        print("COMPUTING FEATURES")

        # Make sure the necessary data has been loaded
        assert ("sample" in self.metadata.keys())

        # read the necessary metadata
        sample = self.metadata["sample"]

        if self.dense:
            self.metadata["features"] = self.model.run_vote_extract_features(sample)
        else:
            self.metadata["features"] = self.model.run_extract_features(sample)

        print("features saved")
        return self.metadata

    def generate_heatmap(self):
        # check if python attack_image.py --image splicing-70.pngthe features have already been processed
        if "heatmap" in self.metadata:
            return self.metadata

        assert ("sample" in self.metadata and self.metadata["sample"] is not None)
        assert ("features" in self.metadata and self.metadata["features"] is not None)

        # read the necessary metadata
        sample = self.metadata["sample"]
        features = self.metadata["features"]

        # compute the heatmap
        heatmap = None
        if self.dense:
            heatmap = self.model.run_vote_cluster(features)[0]
        else:
            heatmap = self.model.run_cluster_heatmap(sample, features, False)

        # Normalize the heatmap between the range [0,1]
        heatmap = (heatmap - heatmap.min())
        heatmap = heatmap / heatmap.max()

        # More than halp of the pixels are above the 0.5 threshold, flip them
        if np.mean(heatmap) > 0.5:
            heatmap = 1 - heatmap

        self.metadata["heatmap"] = heatmap

        return self.metadata

    def generate_mask(self, threshold=None, gt_mask=None, metric=None):
        """
        Function using the extracted metdata to compute the binary mask of the forged area
        @param threshold: threshold to use to segment the heatmap
        @param gt_mask: ground truth mask used to compute the optimal threshold (works only if metric != None)
        @param metric: metric function used to compute the optimal threshold (works only if gt_mask != None)
        @return: mask of the forged area
        """
        mask = None

        assert (threshold is not None or (gt_mask is not None and metric is not None))

        if threshold is not None:

            assert ("heatmap" in self.metadata and self.metadata["heatmap"] is not None)
            print(f"Thresholding at: {threshold}")
            heatmap = self.metadata["heatmap"]
            mask = segment_heatmap(heatmap, threshold)
        elif gt_mask is not None and metric is not None:

            assert ("heatmap" in self.metadata and self.metadata["heatmap"] is not None)
            heatmap = self.metadata["heatmap"]
            print("Finding best threshold using the gt mask")
            print(heatmap.min(),heatmap.max())
            mask = find_optimal_mask(heatmap.copy(), gt_mask, metric)
        else:

            assert ("sample" in self.metadata and self.metadata["sample"] is not None)
            assert ("features" in self.metadata and self.metadata["features"] is not None)

            # read the necessary metadata
            sample = self.metadata["sample"]
            features = self.metadata["features"]

            print("using unsupervised method")

            mask = self.model.run_cluster_mask(sample, features)
        mask = np.array(np.rint(mask), dtype=np.uint8)

        if np.mean(mask == 1) > 0.5:
            mask = 1 - mask

        self.metadata["mask"] = mask

        return mask
