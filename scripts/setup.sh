#!/bin/bash
# setup.sh
# This script sets up the Python virtual environment and necessary directories for the project.
# It should be run from the project's root directory.

# Stop on the first sign of trouble
set -e

# Set environment variables for the project directories
# These variables are used by the Python application to determine where to read/write data, logs, and models.
export PROJECT_ROOT="$(pwd)"
export DATA_DIR="$PROJECT_ROOT/data"
export MODELS_DIR="$PROJECT_ROOT/models"
export LOGS_DIR="$PROJECT_ROOT/logs"

# Print the directories being used (useful for debugging)
echo "Project Root Directory: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Models Directory: $MODELS_DIR"
echo "Logs Directory: $LOGS_DIR"

# Create directories if they don't exist
# These directories are where your project will store its data, models, and logs.
mkdir -p "$DATA_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$LOGS_DIR"
echo "Required directories created."

# Set up the Python virtual environment and install dependencies
# It's important to use a virtual environment to manage dependencies specific to this project.
echo "Setting up the Python virtual environment..."
python -m venv venv
source venv/bin/activate
echo "Virtual environment activated."

# Install Python dependencies from requirements.txt
# These dependencies include all the necessary Python packages for the project to run.
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Dependencies installed."

# Additional setup steps can be added below
# ...

# Inform the user that the setup has completed successfully.
echo "Setup completed successfully."

