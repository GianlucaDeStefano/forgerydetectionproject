from pathlib import Path
from Models.SingleBranchFCNN import SingleBranchFCNN
from Datasets.CASIA2 import CASIA2
import tensorflow_datasets as tfds
from Geneartors.Casia2Generator import Casia2Generator
from tensorflow.python.client import device_lib


#print device information to see if tensorflow is running on the cpu or the gpu
#print(device_lib.list_local_devices())

#get a reference to the CASIA2 dataset, downloading it if not already present
dataset = CASIA2()
dataset.download_and_prepare()

#prepare the training data generator
train_set = dataset.as_dataset(split="train",as_supervised=True)
train_generator = Casia2Generator(train_set, batch_size=20)

#prepare the validation data generator
validation_set = dataset.as_dataset(split="validation",as_supervised=True)
validation_generator = Casia2Generator(train_set, batch_size=20)

#define the model to use
model = SingleBranchFCNN("Simple CNN", Path("Logs"))
model.train_model(train_generator,validation_generator,30,"sparse_categorical_crossentropy")