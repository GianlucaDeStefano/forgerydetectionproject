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
./Datasets/get_datasets.sh
```

## Usage
To analyze an image use the python analyze_image.py script.
```bash
    python analyze_image.py -i <input image path> -g <ground truth image path> -q <quality factor>
```

Use the generate_targets.py script to generate the average target representation for each quality model
```bash
    python generate_targets.py 
```

To attack an image use the following command
```bash
    python attack_image.py -i <input image path> -g <ground truth path> -d
```
In the folder /Data/Debug/<run id>/ the script will save the final comparison and other images useful for debug purposes