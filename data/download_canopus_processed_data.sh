export_link="https://zenodo.org/records/20671842/files/canopus_aug_filt.tar.gz"

mkdir -p data/
cd data/
wget $export_link

tar -xvf canopus_aug_filt.tar.gz
rm -f canopus_aug_filt.tar.gz
cd ../
