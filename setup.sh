#!/usr/bin/bash
# 
# Script to create additional folders, clone data from github repos.  
# 
# W. Probert, 2019

mkdir -p data graphics

# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install numpy pandas matplotlib pytest wheel

# Install the voting_systems package (developer mode)
pip install -e .

# Clone the data from the git repo
git clone git@github.com:p-robot/objectives_matter.git objectives_matter
cp ./objectives_matter/data/*.csv data
rm -rf objectives_matter

# Clean the data on the FMD case study
python3 clean_fmd_case_study_data.py "data" "data"

# Clean the data on the Ebola case study
python3 clean_ebola_case_study_data.py "data" "data"

# Save module information
pip freeze > requirements.txt

# Deactivate the virtual environment
deactivate
