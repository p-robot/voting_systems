#!/usr/bin/bash
# 
# Script to set up the repo - create folders and pull data.  
# 
# W. Probert, 2019

mkdir -p data graphics

# Clone the data from the git repo
git clone git@github.com:p-robot/objectives_matter.git objectives_matter
cp ./objectives_matter/data/*.csv ./data/
rm -rf objectives_matter

# Clean the data on the Ebola case study
# <To add>
