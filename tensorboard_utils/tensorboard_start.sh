#!/bin/bash

# Variables
# Venv directory
VENV_DIR="./tensorboard_venv"
# Tensorboard log directory
LOG_DIR="log"
DATASET_TYPE="clean_road"
METHOD="recon"
TENSORBOARD_LOG_DIR="../${LOG_DIR}/${DATASET_TYPE}/${METHOD}/"

# If given one parameter, use it as the tensorboard log directory
if [ $# -eq 1 ]; then
    TENSORBOARD_LOG_DIR=$1
    TENSORBOARD_LOG_DIR=$(realpath $TENSORBOARD_LOG_DIR)
fi

# Move to the directory of the script
cd "$(dirname "$0")"

# If the virtual environment does not exist, create it and install using tensorboard_venv_requirements.txt
if [ ! -d $VENV_DIR ]; then
    echo "Virtual environment does not exist."
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install -r tensorboard_venv_requirements.txt
    deactivate
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Run the Python command
echo "Starting TensorBoard..."
echo "Log directory: $TENSORBOARD_LOG_DIR"
tensorboard --logdir=$TENSORBOARD_LOG_DIR

# Deactivate the virtual environment
deactivate
