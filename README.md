# LOTS-implementation

This repo contains an implementation of the LTS attack against the noiseprint detector.

## Installation

Clone the project:
```bash
git clone https://github.com/Tesi-magistrale-De-Stefano-Gianluca/LOTS-Noiseprint-implementation.git
cd LOTS-Noiseprint-implementation
```

Create environment (using conda), installing also CUDA:
```bash
conda create --name LOTS-attack-env python=3.5 
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

Use the generate_target.py script to generste the target average noiseprint for the selected quality model. 
In this example we are generating the average noiseprint for the model with quality factor = 101, on the columbia dataset. 
The average noiseprint is then saved with name "avgNoiseprint" in a .npy file
```bash
python generate_target.py -o avgNoiseprint -d columbia -q 101
```