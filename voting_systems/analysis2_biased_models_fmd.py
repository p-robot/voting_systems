#!/usr/bin/env python3
"""
Investigating the effect of adding biased models

Add an additional model with all 1's for first action
(randomly allocating other preferences)

W. Probert, 2020
"""

from os.path import join
import os, sys
from datetime import datetime
import numpy as np, pandas as pd

import voting_systems as voting

vote_processing_rules = [\
    voting.fpp, \
    voting.borda_count, \
    voting.coombs_method, \
    voting.alternative_vote]

dataset_files = [\
    "fmd_data_votes_cattle_culled.csv", \
    "fmd_data_votes_livestock_culled.csv", \
    "fmd_data_votes_duration.csv"]

N_BIASED_MODELS = 5
CANDIDATES = np.arange(5)
Nvotes = 100

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
    for i, filename in enumerate(dataset_files):
        
        # Load datasets
        df_votes = pd.read_csv(join(DATA_DIR, filename))
        
        # Add up to N_BIASED_MODELS to the current votes
        for N in np.arange(0, N_BIASED_MODELS + 1):
            
            # Have the new biased models vote for a single action
            for BIASED_CANDIDATE in CANDIDATES:
                
                vote_cols = [c for c in df_votes.columns if "rank" in c]
                votes = df_votes[vote_cols].to_numpy().astype(int)
                
                # Add an additional model with all 1st preferences for first action
                # (randomly allocating other preferences)
                remaining_candidates = [c for c in CANDIDATES if c != BIASED_CANDIDATE]
                
                if N > 0:
                    new_model_prefs = [[BIASED_CANDIDATE] + \
                    list(np.random.permutation(remaining_candidates)) for i in range(Nvotes * N)]
                    
                    votes = np.vstack((votes, new_model_prefs))
                
                # Process the votes for all vote-processing rule
                for rule in vote_processing_rules:
                    
                    # Run vote-processing rule
                    (winner, winner_index), (candidates, output) = rule(votes)
                    
                    # Save outputs in an array
                    results.append([filename, rule.__name__, N, BIASED_CANDIDATE, winner, now])
    
    # Coerce array to dataframe
    df_results = pd.DataFrame(results)
    df_results.columns = [\
        "dataset", "vote_processing_rule", "number_biased_models", \
        "biased_candidate", "winner", "time"]
    
    df_results = df_results.sort_values(by = ["dataset", "vote_processing_rule",\
        "biased_candidate", "number_biased_models"])
    
    # Output dataframe to output folder
    df_results.to_csv(join(OUTPUT_DIR, "results_analysis2_fmd.csv"), index = False)
