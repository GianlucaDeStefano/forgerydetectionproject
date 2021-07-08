
if [ ! -d "./Data/Datasets/" ]
then
  mkdir  -p "./Data/Datasets/"
fi

# Get Columbia dataset ./Columbia/Data
echo "Preparing Columbia Dataset"
if [ ! -d "./Data/Datasets/Columbia" ]
then
wget -O ImSpliceDataset.rar "https://www.dropbox.com/s/bo10et4p1zg08aj/ImSpliceDataset.rar?dl=1"
unrar x ImSpliceDataset.rar ./Data/Datasets/Columbia/
mv ./Data/Datasets/Columbia/ImSpliceDataset ./Data/Datasets/Columbia
rm ImSpliceDataset.rar
fi

echo "Preparing Columbia uncompressed Dataset"
if [ ! -d "./Data/Datasets/ColumbiaUncompressed" ]
then
wget -O 4cam_auth.tar.bz2 "https://www.dropbox.com/sh/786qv3yhvc7s9ki/AABaQvI-lPiM3Zl64RQoDCiMa/4cam_auth.tar.bz2?dl=1"
wget -O 4cam_splc.tar.bz2 "https://www.dropbox.com/sh/786qv3yhvc7s9ki/AAAESATxO7wncDMKkl1XjyNaa/4cam_splc.tar.bz2?dl=1"
mkdir  -p "./Data/Datasets/ColumbiaUncompressed"
tar -xvjf  4cam_auth.tar.bz2 -C ./Data/Datasets/ColumbiaUncompressed
tar -xvjf 4cam_splc.tar.bz2 -C ./Data/Datasets/ColumbiaUncompressed
rm 4cam_auth.tar.bz2
rm 4cam_splc.tar.bz2
fi


echo " - Columbia uncompressed dataset is ready"

#download the realistic image tampering dataset
echo "Preparing 'realistic image tampering'"
if [ ! -f "./Data/Datasets/RIT/readme.md" ]
then
gdown https://drive.google.com/u/0/uc?id=0B73Fq3C_nT4aOThud0NYWUR2MTQ -O realistic-tampering-dataset.zip
unzip -qq realistic-tampering-dataset.zip -d .
mkdir -p ./Data/Datasets/RIT/
mv ./data-images/* ./Data/Datasets/RIT/
rm realistic-tampering-dataset.zip
rm -r ./data-images/
fi

echo "Datasets setup successfully"

echo "Preparing DSO-1 Dataset"
if [ ! -f "./Data/Datasets/DSO/Masks" ]
then
  if [ ! -f "./tifs-database.zip" ]
  then
    echo "Downloading the dataset"
    wget -O ./tifs-database.zip "http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip"
  fi
unzip -qq ./tifs-database.zip -d .
mkdir -p ./Data/Datasets/DSO/Data/
mv ./tifs-database/DSO-1 ./Data/Datasets/DSO/images
mv ./tifs-database/DSO-1-Fake-Images-Masks ./Data/Datasets/DSO/masks
rm -r ./tifs-database/
rm -r ./__MACOSX/
rm ./tifs-database.zip
fi