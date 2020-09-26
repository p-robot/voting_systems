#!/usr/bin/env python3
"""
Script for running vote processing rules on Ebola and FMD case studies.  

Usage
-----




W. Probert, 2020
"""

import numpy as np, pandas as pd, os, sys
from os.path import join
from datetime import datetime

import voting_systems as voting

vote_processing_rules = [\
    voting.fpp, \
    voting.borda_count, \
    voting.coombs_method, \
    voting.alternative_vote]

dataset_files = [\
    "ebola_data_votes_str_cases.csv", \
    "fmd_data_votes_str_cattle_culled.csv", \
    "fmd_data_votes_str_livestock_culled.csv", \
    "fmd_data_votes_str_duration.csv"]

if __name__ == "__main__":
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        DATA_DIR = "data"

    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    else: 
        OUTPUT_DIR = "."
    
    # Pull today's date (for saving in output files)
    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    
    # List for storing results
    results = []
    
    # Loop through datasets
    for filename in dataset_files:
        
        # Load dataset
        df_votes = pd.read_csv(join(DATA_DIR, filename))
        
        vote_cols = [c for c in df_votes.columns if "rank" in c]
        
        votes = df_votes[vote_cols].to_numpy().astype(str) # to avoid 'object' type
        
        # Process 'election' for each vote-processing rule
        for rule in vote_processing_rules:
            
            # Run vote-processing rule
            (winner, winner_index), (candidates, output) = rule(votes)
            
            # Save outputs in an array
            results.append([filename, rule.__name__, winner, now])
    
    # Coerce array to dataframe
    df_results = pd.DataFrame(results)
    df_results.columns = ["dataset", "vote_processing_rule", "winner", "time"]
    
    # Output dataframe to output folder
    df_results.to_csv(join(OUTPUT_DIR, "results_analysis1.csv"), index = False)
