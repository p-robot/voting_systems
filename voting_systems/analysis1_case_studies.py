#!/usr/bin/env python3
"""
Script for running analysis of Ebola and FMD data.  

To do
------

FMD cleaning script: Also output wide dataset of FMD output.  
FMD cleaning script: Also output dataset of FMD with "control" categories instead of rankings.  
FMD cleaning script: Add columns to FMD datasets (add model_id and run_id to column names, remove any columns with "_id" in their name).  


Data cleaning: Add "objective" to csv datasets (perhaps in filename or in comments at start of file)
Data cleaning: Add "name" to csv datasets (perhaps in filename)
Data cleaning: Check FMD and Ebola examples are cleaning the vote data in the same way.  


Output: Check that outputted winners are correct.  


W. Probert, 2020
"""

import numpy as np, pandas as pd
import voting_systems.core as voting
from os.path import join
import os

DATA_DIR = "voting_systems/data"

vote_processing_rules = [\
    voting.fpp, \
    voting.borda_count, \
    voting.coombs_method, \
    voting.alternative_vote]

dataset_files = ["ebola_data_votes_cases.csv", \
    "fmd_data_votes_cattle_culled.csv", \
    "fmd_data_votes_livestock_culled.csv", \
    "fmd_data_votes_duration.csv"]

if __name__ == "__main__":
    
    # List for storing results
    results = []
    
    # Loop through datasets
    for filename in dataset_files:
        
        # Load datasets
        df_votes = pd.read_csv(join(DATA_DIR, filename))
        
        if "ebola" in filename:
            idx = 1
        else:
            idx = 2
        
        votes = df_votes.to_numpy()[:, idx:]
        
        for rule in vote_processing_rules:
            
            # Run vote-processing rule
            (winner, winner_index), (candidates, output) = rule(votes)
            
            # Save outputs in an array
            results.append([filename, rule.__name__, winner])
        
    # Coerce array to dataframe
    df_results = pd.DataFrame(results)
    df_results.columns = ["dataset", "vote_processing_rule", "winner"]
    
    # Output dataframe to output folder
    df_results.to_csv(join("results", "results_analysis1.csv"), index = False)
    
    #"analysis1.csv"
    # today's date
    # log file

