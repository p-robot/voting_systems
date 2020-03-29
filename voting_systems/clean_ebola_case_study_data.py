#!/usr/bin/env python3
"""
Process the output from Li et al (2017) PNAS.


This script pulls the data forwarded from Shouli, converts the performance scores into ranked votes, and runs the vote-processing rules on the data (as defined within voting_systems.py).  

W. Probert, 2019
"""

import pandas as pd, numpy as np, re
from os.path import join

# Read in excel sheet.  
df = pd.read_csv(join("voting_systems", \
    "data", "Ebola caseload under five interventions_Shouli.csv"))

# Replace all white space in names with underscore
# (after first removing trailing and leading white space)
df.columns = [re.sub(r"\s+", "_", c.lower().rstrip()) for c in df.columns]

actions = df.columns

df.columns = ['model', 'a', 'b', 'c', 'd', 'e', 'f']

# Tidy model names (remove semicolon, remove white space, make lower case)
df['model'] = df['model'].str.replace(" ", "")
df['model'] = df['model'].str.replace(";", "")
df['model'] = df['model'].str.lower()

df.to_csv(join("voting_systems", "data", "ebola_data_cleaned.csv"), index = False)

# Extract only model names
models = df.model.values

# Generate votes for each action from each model
votes = [np.argsort(row.values[1:]) for (index, row) in df.iterrows()]
votes = np.asarray(votes)
#pd.value_counts([np.argmin(d[1].values[1:]) for d in  df.iterrows()])/37

df_votes = pd.DataFrame(np.append(models[:,None], votes, axis = 1), 
    columns = ["model", "a", "b", "c", "d", "e", "f"])

df_votes.to_csv(join("voting_systems", "data", "ebola_data_votes_cases.csv"), index = False)

