# %%

from pathlib import Path
import matplotlib.pyplot as plt
from Datasets.CASIA2 import CASIA2
from Datasets.Utilities.Maps.Noiseprint.noiseprint import normalize_noiseprint

# Download and prepare the dataset
# This will take a while since we have to process each image singularly to extract the noise features
from Geneartors.CASIA2.Casia2Generator import Casia2Generator
from Models.Customs.ClassifierBase import ClassifierBase

dataset = CASIA2()
dataset.download_and_prepare()



train_split = dataset.as_dataset(split="train")

n_cols = 4

nsamples = 5

samples = train_split.take(nsamples)

col_titles = ['Image', 'Noiseprint', 'SRM']

nrows = nsamples
ncols = len(col_titles)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, 18))  # create the figure with subplots
[ax.set_axis_off() for ax in axes.ravel()]  # remove the axis

for ax, col in zip(axes[0], col_titles):  # set up the title for each column
    ax.set_title(col, fontdict={'fontsize': 18, 'color': 'b'})

i = 0
for sample in samples:
    axes[i, 0].imshow(sample["image"])
    axes[i, 1].imshow(normalize_noiseprint(sample["noiseprint"].numpy()))
    axes[i, 2].imshow(sample["SRM"])
    # axes[i,3].imshow(sample['mask'])

    i = i + 1


# Define parameters essentials for the training of the models

# Define input parameters
input_shape_rgb = (256, 384, 3)
input_shape_rbf = (256, 384, 3)
input_shape_noiseprint = (256, 384, 1)

# We just have to distinguish between tampered and pristine images
# and a single class is enough for that
output_classes = 1

# Define the loss the models will use
loss_function = "binary_crossentropy"

# Define the number of epochs each model has to be trained for
epochs = 30

# define the size of each training batch
batch_size = 16


# Define additional parameters not essentials for the training

# Set the path to the Log folder in which the logs, the checkpoints and other usefull
# data will be used
logs_folder = Path("./Logs")

# Set verbose = True if you want an extensive printing of logs during the training
# and testing of the models
verbose = True


# Create 2 generator of datas that has that provide samples with the following structure:
#   X -> [RGB image]
#   Y -> class of the image
# The first generator will produce training data, the second will produce validation data

generator_training_rgb = Casia2Generator(dataset.as_dataset(split="train"), ["rgb"], batch_size)
generator_validation_rgb = Casia2Generator(dataset.as_dataset(split="validation"), ["rgb"], batch_size)
# Train a Resnet Classifier using the RGB data
model_rgb = ClassifierBase(input_shape_rgb, output_classes, "RGB model", logs_folder, verbose)
model_rgb.train_model(generator_training_rgb, generator_validation_rgb, epochs, loss_function, save_model=True)


