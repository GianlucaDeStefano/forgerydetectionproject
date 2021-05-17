# Get Columbia dataset ./Columbia/Data
wget -O ImSpliceDataset.rar "https://www.dropbox.com/s/bo10et4p1zg08aj/ImSpliceDataset.rar?dl=1"
unrar x ImSpliceDataset.rar ./Columbia/
mv ./Columbia/ImSpliceDataset ./Columbia/Data
rm ImSpliceDataset.rar
