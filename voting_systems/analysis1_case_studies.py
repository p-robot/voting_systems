#!/usr/bin/env python3
"""
Script for running analysis of Ebola and FMD data

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

datasets_files = ["ebola_data_votes.csv"]#, \
#    "fmd_data_votes_culls.csv", \
#    "fmd_data_votes_duration.csv"]


if __name__ == "__main__":
    
    # Load datasets
    datasets = [pd.read_csv(join(DATA_DIR, filename)) for filename in datasets_files]
    
    # Loop through datasets
    for df_votes in datasets:
        
        votes = df_votes.to_numpy()[:, 1:]
    
        for rule in vote_processing_rules:
            print(rule.__name__)
            # Run vote-processing rule
            (winner, winner_index), (candidates, output) = rule(votes)
            
            print(winner)
            
            # Save outputs in an array
            #results.append([rule.name, winner])
    
        # Coerce array to dataframe
    
        # Output dataframe to output folder
        #"analysis1.csv"
        # today's date
        # log file

