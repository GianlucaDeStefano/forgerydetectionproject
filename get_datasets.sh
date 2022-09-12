
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

echo "Preparing DSO-1 Dataset"
if [ ! -f "./Data/Datasets/DSO/masks" ]
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

echo "Preparing Detectors"

if [ ! -f "./Detectors/Exif/ckpt" ]
then
echo "Downloading exif_final.zip"
  # Google Drive link to exif_final.zip
  gdown https://drive.google.com/uc?id=1X6b55rwZzU68Mz1m68WIX_G2idsEw3Qh -O exif_final.zip

  mkdir -p ./Detectors/Exif/ckpt/
  unzip exif_final.zip -d ./Detectors/Exif/ckpt/

  rm exif_final.zip
fi