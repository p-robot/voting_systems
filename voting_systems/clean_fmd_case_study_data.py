#!/usr/bin/env python3
"""
Process the output from Probert et al (2016)

This script pulls the data from Probert et al. (2016), converts the performance scores into ranked
votes, and runs the vote-processing rules on the data (as defined within voting_systems.py).  

W. Probert, 2020
"""

import pandas as pd, numpy as np, re
from os.path import join

# Read in CSV files
models = ["a", "b", "c", "d", "e"]
datasets = [pd.read_csv(join("voting_systems", "data", "model_" + m + ".csv")) for m in models]
df = pd.concat(datasets)

controls = df.control.unique()
df['livestock_culled'] = df.cattle_culled + df.sheep_culled
objectives = ['duration', 'livestock_culled', 'cattle_culled']


# Function = Values2Votes(df, objective_col = "duration" objective_fn = "min", id_cols, direction = "wide", control_col = None)
# With direction "wide" the column names are the controls.  With direction "long" control names are in the column "control"
# can save wide format, values, votes, ... 
# EBOLA - to long format.  then use this function


for obj in objectives:
    df_wide = df.groupby(['model', 'run', 'control'])[obj].sum().unstack('control').reset_index()
    print(df_wide.head())
    df_wide["objective"] = obj
    
    df_wide.to_csv(join("voting_systems", "data", "fmd_data_cleaned_"+obj+".csv"), index = False)
    
    # Generate votes for each action from each model
    votes = [np.argsort(row.values[2:]) for (index, row) in df_wide.iterrows()]
    votes = np.asarray(votes)
    
    # Extract only model names
    models = df.model.values
    
    df_votes = pd.DataFrame(np.append(df_wide[['model', 'run']].values, votes, axis = 1))
    df_votes.to_csv(join("voting_systems", "data", "fmd_data_votes_" + obj + ".csv"), index = False)
