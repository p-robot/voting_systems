#!/usr/bin/env python3
"""
Process the output from Probert et al (2016)

This script does the following: 
1. reads data from Probert et al. (2016)
2. converts the performance scores into ranked votes

W. Probert, 2020
"""

import pandas as pd, numpy as np, re, sys
from os.path import join
import voting_systems as voting

if __name__ == "__main__":
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        DATA_DIR = "data"

    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    else: 
        OUTPUT_DIR = "data"
    
    # Set seed for repeatable results
    np.random.seed(2019)
    
    # Read in CSV files
    models = ["a", "b", "c", "d", "e"]
    datasets = [pd.read_csv(join(DATA_DIR, "model_" + m + ".csv")) for m in models]
    df = pd.concat(datasets)

    actions = df.control.unique()
    df['livestock_culled'] = df.cattle_culled + df.sheep_culled

    # Primary objective for ranking actions
    objectives = ['duration', 'livestock_culled', 'cattle_culled']
    
    # Secondary objective for ranking actions when ties occur in the primary objective
    secondary_objectives = ['cattle_culled', 'duration', 'duration']
    
    for obj, sec_obj in zip(objectives, secondary_objectives):
        group_vars = ['model', 'run', 'control']
        df_wide = df.groupby(group_vars)[obj].sum().unstack('control').reset_index()
        df_wide["objective"] = obj
        
        df_wide.to_csv(join(OUTPUT_DIR, "fmd_data_cleaned_" + obj + ".csv"), index = False)
        
        values = df_wide[actions].to_numpy()
        
        # Secondary objective
        df_sec_obj = df.groupby(group_vars)[sec_obj].sum().unstack('control').reset_index()
        sec_obj_values = df_sec_obj[actions].to_numpy()
        
        # Generate votes for each action from each model (returning indices)
        votes = voting.values_to_votes(values, secondary_value = [sec_obj_values])
        
        # Generate votes for each action from each model (returning action labels)
        votes_str = voting.values_to_votes(values, 
            secondary_value = [sec_obj_values], candidate_labels = actions)
        
        # Extract only model names
        models = df.model.values
        
        colnames = ['model', 'run'] + [f'rank{i}' for i in np.arange(1, len(actions)+1)]
        
        # Rearrange columns of data frames and save to file
        df_votes = pd.DataFrame(np.append(df_wide[['model', 'run']].values, votes, axis = 1))
        df_votes.columns = colnames
        df_votes.insert(2, "objective", "minimize " + obj.replace("_", " "))
        df_votes.to_csv(join(OUTPUT_DIR, "fmd_data_votes_" + obj + ".csv"), index = False)
    
        df_votes_str = pd.DataFrame(np.append(df_wide[['model', 'run']].values, 
            votes_str, axis = 1))
        df_votes_str.columns = colnames
        df_votes_str.insert(2, "objective", "minimize " + obj.replace("_", " "))
        df_votes_str.to_csv(join(OUTPUT_DIR, "fmd_data_votes_str_" + obj + ".csv"), index = False)
