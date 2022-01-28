# Forgery detection project
This is a project for the course of Machine Learning in Cyber Security. It's aim is to produce a machine learning model capable of identifying tampered regions of images testin also its performances against black box adversarial attacks.

## Installation

Clone this repo:
```bash
git clone https://github.com/GianlucaDeStefano/forgerydetectionproject.git
```

Create a conda environment to run it:

```
conda create --name tf_gpu tensorflow-gpu 
```

Activate the environment:
```
conda activate tf_gpu 
```


Install  the requirements using pip:

```
pip install -r requirements.txt
```

## Usage

To train a model execute:

```bash
python train.py
```

## Folder structure

    .
    ├── Datasets                # Contains the builder classes to download and save datasets
    ├── Generators              # Contains the generator classes to load datasets from memory
    ├── Logs                    # Folder used to save log data during training
    ├── Models                  # Contains the implementation of the classes to build and train the model
    ├── train.py                # Main endpoint to use to train a model
    └── README.md


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
