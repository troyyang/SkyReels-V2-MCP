#!/bin/bash

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv (https://github.com/astral-sh/uv) first."
    exit 1
fi

# Clone repository
echo "Cloning SkyReels-V2 repository..."
if ! git clone https://github.com/SkyworkAI/SkyReels-V2; then
    echo "Error: Failed to clone SkyReels-V2 repository."
    exit 1
fi

# Verify clone was successful
if [ ! -d "SkyReels-V2" ]; then
    echo "Error: Cloned directory SkyReels-V2 not found."
    exit 1
fi

# Copy all files including hidden ones
echo "Copying files..."
cp -r SkyReels-V2/. .

# Install dependencies
echo "Installing dependencies..."
uv sync || { echo "Error: uv sync failed"; exit 1; }

# Install additional packages
echo "Installing torch..."
uv pip install torch || { echo "Error: torch installation failed"; exit 1; }

echo "Installing flash-attn..."
uv pip install flash-attn --no-build-isolation || { echo "Error: flash-attn installation failed"; exit 1; }

echo "Installation completed successfully!"