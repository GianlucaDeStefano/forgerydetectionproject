from datasets.CASIA2 import CASIA2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
#import dataset downloading it if necessary
from generators.Casia2Generator import Casia2Generator

dataset = CASIA2()
dataset.download_and_prepare()



ds = tfds.load('CASIA2',split="train",as_supervised=True)
generator = Casia2Generator(ds,32,True)