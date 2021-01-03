from pathlib import Path

from Models.CNNModel import CNNModel
from Models.SingleBranchFCNN import SingleBranchFCNN
from datasets.CASIA2 import CASIA2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
#import dataset downloading it if necessary
from generators.Casia2Generator import Casia2Generator


#get a reference to the CASIA2 dataset, downloading it if not already present
dataset = CASIA2()
dataset.download_and_prepare()

#prepare the training data generator
train_set = tfds.load('CASIA2',split="train",as_supervised=True)
train_generator = Casia2Generator(train_set, batch_size=32)

#prepare the validation data generator
validation_set = tfds.load('CASIA2',split="validation",as_supervised=True)
validation_generator = Casia2Generator(train_set, batch_size=32)

#define the model to use
model = SingleBranchFCNN("Simple CNN",Path("./logs"),(None,None,3),(None,None,1))
model.train_model(train_generator,validation_generator,30)