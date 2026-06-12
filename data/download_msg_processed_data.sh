export_link="https://zenodo.org/records/20671842/files/msg_aug_filt.tar.gz"

mkdir -p data/
cd data/
wget $export_link

tar -xvf msg_aug_filt.tar.gz
rm -f msg_aug_filt.tar.gz
cd ../
