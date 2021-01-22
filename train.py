from pathlib import Path
from Datasets.CASIA2 import CASIA2
from Geneartors.CASIA2.Casia2Generator import Casia2RGBGenerator


#print device information to see if tensorflow is running on the cpu or the gpu
#print(device_lib.list_local_devices())

#get a reference to the CASIA2 dataset, downloading it if not already present
from Models.Customs.Unet import Unet

dataset = CASIA2()
dataset.download_and_prepare()

#prepare the training data generator
train_set = dataset.as_dataset(split="train")
train_generator = Casia2RGBGenerator(train_set,["rgb"], batch_size=20)

#prepare the validation data generator
validation_set = dataset.as_dataset(split="validation")
validation_generator = Casia2RGBGenerator(validation_set,["rgb"], batch_size=20)

#define the model to use
#model = Unet("Simple CNN", Path("Logs"))
#model.train_model(train_generator,validation_generator,30,"binary_crossentropy")