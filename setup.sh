#!/usr/bin/bash
# 
# Script to create additional folders, clone data from github repos.  
# 
# W. Probert, 2019

mkdir -p data graphics

# Clone the data from the git repo
git clone git@github.com:p-robot/objectives_matter.git objectives_matter
cp ./objectives_matter/data/*.csv ./data/
rm -rf objectives_matter

# Clean the data on the FMD case study
python3 clean_fmd_case_study_data.py

# Clean the data on the Ebola case study
python3 clean_ebola_case_study_data.py
