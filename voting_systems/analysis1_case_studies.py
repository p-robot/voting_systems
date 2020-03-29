#!/usr/bin/env python3
"""
Script for running vote processing rules on Ebola and FMD case studies.  

W. Probert, 2020
"""

from os.path import join
import os, sys
from datetime import datetime
import numpy as np, pandas as pd

import voting_systems.core as voting

vote_processing_rules = [\
    voting.fpp, \
    voting.borda_count, \
    voting.coombs_method, \
    voting.alternative_vote]

dataset_files = [\
    "ebola_data_votes_cases.csv", \
    "fmd_data_votes_cattle_culled.csv", \
    "fmd_data_votes_livestock_culled.csv", \
    "fmd_data_votes_duration.csv"]

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
    
    print(DATA_DIR)
    print(OUTPUT_DIR)
    
    # Loop through datasets
    for filename in dataset_files:
        
        # Load datasets
        df_votes = pd.read_csv(join(DATA_DIR, filename))
        
        if "ebola" in filename:
            idx = 1
        else:
            idx = 2
        
        votes = df_votes.to_numpy()[:, idx:].astype(int)
        
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
