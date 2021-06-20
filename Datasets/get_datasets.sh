
# Get Columbia dataset ./Columbia/Data
echo "Preparing Columbia Dataset"
if [ ! -d "./Columbia/Data" ]
then
wget -O ImSpliceDataset.rar "https://www.dropbox.com/s/bo10et4p1zg08aj/ImSpliceDataset.rar?dl=1"
unrar x ImSpliceDataset.rar ./Columbia/
mv ./Columbia/ImSpliceDataset ./Columbia/Data
rm ImSpliceDataset.rar
fi

echo "Preparing Columbia uncompressed Dataset"
if [ ! -d "./ColumbiaUncompressed/Data" ]
then
wget -O 4cam_auth.tar.bz2 "https://www.dropbox.com/sh/786qv3yhvc7s9ki/AABaQvI-lPiM3Zl64RQoDCiMa/4cam_auth.tar.bz2?dl=1"
wget -O 4cam_splc.tar.bz2 "https://www.dropbox.com/sh/786qv3yhvc7s9ki/AAAESATxO7wncDMKkl1XjyNaa/4cam_splc.tar.bz2?dl=1"
mkdir  -p "./ColumbiaUncompressed/Data"
tar -xvjf  4cam_auth.tar.bz2 -C ./ColumbiaUncompressed/Data/
tar -xvjf 4cam_splc.tar.bz2 -C ./ColumbiaUncompressed/Data/
rm 4cam_auth.tar.bz2
rm 4cam_splc.tar.bz2
fi


echo " - Columbia uncompressed dataset is ready"

#download the realistic image tampering dataset
echo "Preparing 'realistic image tampering'"
if [ ! -f "./RIT/Data/readme.md" ]
then
gdown https://drive.google.com/u/0/uc?id=0B73Fq3C_nT4aOThud0NYWUR2MTQ -O realistic-tampering-dataset.zip
unzip -qq realistic-tampering-dataset.zip -d .
mkdir -p ./RIT/Data/
mv ./data-images/* ./RIT/Data/
rm realistic-tampering-dataset.zip
rm -r ./data-images/
fi

echo "Datasets setup successfully"

echo "Preparing DSO-1 Dataset"
if [ ! -f "./DSO/Data/Masks" ]
then
  if [ ! -f "./tifs-database.zip" ]
  then
    echo "Downloading the dataset"
    wget -O ./tifs-database.zip "http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip"
  fi
unzip -qq ./tifs-database.zip -d .
mkdir -p ./DSO/Data/
mv ./tifs-database/DSO-1 ./DSO/Data/images
mv ./tifs-database/DSO-1-Fake-Images-Masks ./DSO/Data/masks
rm -r ./tifs-database/
rm -r ./__MACOSX/
rm ./tifs-database.zip
fi