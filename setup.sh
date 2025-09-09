#!/bin/bash
# DigitalOcean setup script

echo "🔧 Setting up Gaia Knowledge Generator on DigitalOcean..."

# Install system dependencies
sudo apt update
sudo apt install -y wget curl python3-pip python3-venv

# Create virtual environment
python3 -m venv /app/venv
source /app/venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download models
chmod +x download-models.sh
./download-models.sh

echo "✅ Setup completed!"