
# Get Columbia dataset ./Columbia/Data
echo "Preparing Columbia Dataset"
if [ ! -d "./Columbia/Data" ]
then
wget -O ImSpliceDataset.rar "https://www.dropbox.com/s/bo10et4p1zg08aj/ImSpliceDataset.rar?dl=1"
unrar x ImSpliceDataset.rar ./Columbia/
mv ./Columbia/ImSpliceDataset ./Columbia/Data
rm ImSpliceDataset.rar
fi
echo " - Columbia Dataset is ready"

#Get the Casia2 Dataset in 2 steps
echo "Preparing Casia2 Dataset"
if [ ! -d "./Casia2/Data/Au" ]
then
#download the images
gdown https://drive.google.com/u/0/uc?id=1IDUgcoUeonBxx2rASX-_QwV9fhbtqdY8 -O Casia2.zip
unzip -qq Casia2.zip -d ./Casia2/Data/
mv  ./Casia2/Data/CASIA2.0_revised/Au ./Casia2/Data/Au
mv  ./Casia2/Data//CASIA2.0_revised/Tp ./Casia2/Data/Tp
rm Casia2.zip
rm -r ./Casia2/Data/CASIA2.0_revised
fi

if [ ! -d "./Casia2/Data/Gt" ]
then
#download the gt maps
wget -O Casia2_gt.zip "https://github.com/namtpham/casia2groundtruth/raw/master/CASIA2.0_Groundtruth.zip"
unzip -qq Casia2_gt.zip
mv "./CASIA2.0_Groundtruth" ./Casia2/Data/Gt/
rm Casia2_gt.zip
fi
echo " - Casia2 Dataset is ready"

#download the realistic image tampering dataset
echo "Preparing 'realistic image tampering'"
if [ ! -f "./RIT/Data/readme.md" ]
then
gdown https://drive.google.com/u/0/uc?id=0B73Fq3C_nT4aOThud0NYWUR2MTQ -O realistic-tampering-dataset.zip
unzip -qq realistic-tampering-dataset.zip -d .
mv ./data-images/* ./RIT/Data/
rm realistic-tampering-dataset.zip
rm -r ./data-images/
fi

echo "Datasets setup successfully"
