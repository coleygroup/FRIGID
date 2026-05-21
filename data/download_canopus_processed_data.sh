# Original data link
#SVM_URL="https://bio.informatik.uni-jena.de/wp/wp-content/uploads/2020/08/svm_training_data.zip"

export_link="TODO: update with new data link"

mkdir -p data/
cd data/
wget $export_link

tar -xvf canopus.tar.gz
rm -f canopus.tar.gz
cd ../
