#!/bin/bash
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Symblink project directory to Colab runtime
ln -s /content/drive/MyDrive/SkinSight ./project

# Install dependecies
pip install -r ./project/requirements.txt

# Initialize SQLite DB for metadata
sqlite3 ./project/data/metadata.db <<EOF
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  filepath TEXT,
  filepath TEXT,
  resolution TEXT,
  diagnosis TEXT
);
EOF
