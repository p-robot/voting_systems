#!/usr/bin/bash
# 
# Script to create additional folders, clone data from github repos.  
# 
# W. Probert, 2019


data_dir="voting_systems/data"
output_dir="results"

# Run analysis 1
python3 voting_systems/analysis1_case_studies.py "$data_dir" "$output_dir"


# Run analysis 2
python3 voting_systems/analysis2_biased_models_fmd.py "$data_dir" "$output_dir"

# Run analysis 3
