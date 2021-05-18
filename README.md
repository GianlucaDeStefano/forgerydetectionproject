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
conda create LOTS-attack-env
conda install -c anaconda cudnn
conda install -c anaconda cudatoolkit
conda install --file requirements.txt
```

To download the datasets: 
```bash
./Datasets/get_datasets.sh
```

Use the generate_target.py script to generste the target average noiseprint for the selected quality model
```bash
python generate_target.py
```