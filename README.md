# LOTS-implementation

This repo contains an implementation of the LTS attack against the noiseprint detector.

## Setup

Clone the project:
```bash
git clone https://github.com/Tesi-magistrale-De-Stefano-Gianluca/LOTS-Noiseprint-implementation.git
cd LOTS-Noiseprint-implementation
```

Create environment (using conda), installing also CUDA:
```bash
conda create --name LOTS-attack-env python=3.8 
conda activate LOTS-attack-env
conda install -c anaconda cudatoolkit
conda install -c anaconda cudnn
pip install -r requirements.txt
```

To download the datasets: 
```bash
cd Datasets
./get_datasets.sh
```

## Usage
The easiest way of testing the attack is the following: 
```bash
python ./attack_image.py -i canong3_canonxt_sub_13.tif
```
Since this image belongs to one of the supported datasets the script automatically finds its mask and loads
it correctly. We can also test other 3rd party images by passing the path to image and mask directly in the following way
```bash
python ./attack_image.py -i <path_to_image> -g <path_to_mask> 
```
In the folder /Data/Debug/<run id>/ the script will save the final comparison and other images useful for debug purposes