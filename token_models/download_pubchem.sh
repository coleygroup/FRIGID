#!/bin/bash

# 1. Define Variables and Folder
DATA_DIR="data"
URL="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
FILE_GZ="${DATA_DIR}/CID-SMILES.gz"
FILE_TXT="${DATA_DIR}/CID-SMILES"

# Create the data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# 2. Download the file into the data folder
echo "Step 1/3: Downloading PubChem data into '$DATA_DIR'..."
if command -v wget &> /dev/null; then
    wget -q --show-progress -O "$FILE_GZ" "$URL"
elif command -v curl &> /dev/null; then
    curl -# -o "$FILE_GZ" "$URL"
else
    echo "Error: Neither 'wget' nor 'curl' found."
    exit 1
fi

# Verify download
if [ ! -f "$FILE_GZ" ]; then
    echo "Error: Download failed."
    exit 1
fi

# 3. Unzip the file
echo "Step 2/3: Unzipping file inside '$DATA_DIR'..."
# gunzip replaces .gz with the uncompressed file in the same directory
gunzip -f "$FILE_GZ"

# 4. Count the structures
echo "Step 3/3: Counting structures..."
if [ -f "$FILE_TXT" ]; then
    COUNT=$(wc -l < "$FILE_TXT")
    FORMATTED_COUNT=$(printf "%'.f" $COUNT)
    
    echo "----------------------------------------"
    echo "Success! All files are in the '$DATA_DIR/' folder."
    echo "Total number of structures: $FORMATTED_COUNT"
    echo "----------------------------------------"
else
    echo "Error: Unzipping failed."
    exit 1
fi
