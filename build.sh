
#!/bin/bash

# Download libtorch built CPU libraries
URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip" # stable version 2.0.1
LIBTORCH_FILE_NAME="libtorch-shared-with-deps-2.0.1+cpu.zip"
BUILD_DIR="build"

# Check if the file exists
if [ ! -f "third_party/$LIBTORCH_FILE_NAME" ]; then
    # If the file doesn't exist, download it
    cd third_party
    wget "$URL" >> libtorch_setup.txt
    # Unzip the file
    unzip "$LIBTORCH_FILE_NAME" >> libtorch_setup.txt
    cd ..
fi

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

pip install . 
cd build
cmake ..
make