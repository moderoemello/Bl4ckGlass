#!/bin/bash

# Navigate to project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source env/bin/activate

# Run your Python automation script
python main.py

