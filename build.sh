
#!/bin/bash

# Download libtorch built CPU libraries
URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip" # stable version 2.0.1
LIBTORCH_FILE_NAME="libtorch-shared-with-deps-2.0.1+cpu.zip"
BUILD_DIR="build"

# Check if the file exists
if [ ! -f "third_party/$LIBTORCH_FILE_NAME" ]; then
    # If the file doesn't exist, download it
    cd third_party
    wget "$URL"
    # Unzip the file
    unzip "$LIBTORCH_FILE_NAME"
    cd ..
fi

download_and_extract(){
    URL=$1
    FILE_NAME=$2
    if [ ! -f "third_party/$FILE_NAME" ]; then
        # If the file doesn't exist, download it
        cd third_party
        wget "$URL"
        # Unzip the file
        tar -xvzf "$FILE_NAME"
        cd ..
    fi

}

# Download OpenFST
URL="https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.8.2.tar.gz"
OPENFST_FILE_NAME="openfst-1.8.2.tar.gz"
download_and_extract "$URL" "$OPENFST_FILE_NAME"



# Download boost
URL="https://github.com/parlance/ctcdecode/releases/download/v1.0/boost_1_67_0.tar.gz"
BOOST_FILE_NAME="boost_1_67_0.tar.gz"
download_and_extract "$URL" "$BOOST_FILE_NAME"


if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

pip install . 
cd build
cmake ..
make