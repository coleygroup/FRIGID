# This script downloads preprocessed data from the MassSpecGym project
# Original MassSpecGym code/data: https://github.com/pluskal-lab/MassSpecGym

export_link="TODO: update with new data link"

mkdir -p data/
cd data/
wget $export_link

tar -xvf msg.tar.gz
rm -f msg.tar.gz
cd ../
