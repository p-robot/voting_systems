#!/usr/bin/env python3
"""
Process data from Li et al (2017) PNAS

This script pulls the data forwarded from Shouli, saves two files: 

* `ebola_data_cleaned.csv`: a "cleaned" version of the original data
* `ebola_data_votes_cases.csv`: a version converted into ranked votes (showing action ranks)
* `ebola_data_votes_str_cases.csv`: a version converted into ranked votes (showing action labels)

The resultant CSV file of votes has votes as rows and columns are preference/rank.  So each row
lists the possible actions in order of preference.  

W. Probert, 2019
"""

import pandas as pd, numpy as np, re, sys
from os.path import join


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

    # Read in excel sheet.  
    df = pd.read_csv(join(DATA_DIR, "Ebola caseload under five interventions_Shouli.csv"))

    # Replace all white space in names with underscore
    # (after first removing trailing and leading white space)
    _, *actions = df.columns
    actions = [re.sub(r"\s+", "_", c.lower().rstrip()) for c in actions]
    actions = [re.sub("hospitlaization", "hospitalization", a) for a in actions]
    actions = [re.sub("motality", "mortality", a) for a in actions]
    actions = [re.sub("%", "pc", a) for a in actions]

    df.columns = ['model'] + actions

    # Tidy model names (remove semicolon, remove white space, make lower case)
    df['model'] = df['model'].str.replace(" ", "")
    df['model'] = df['model'].str.replace(";", "")
    df['model'] = df['model'].str.lower()

    df.to_csv(join(OUTPUT_DIR, "ebola_data_cleaned.csv"), index = False)

    # Extract only model names
    models = df.model.values

    # Generate votes for each action from each model
    votes = [np.argsort(row.values[1:]) for (index, row) in df.iterrows()]
    votes = np.asarray(votes)

    df_votes = pd.DataFrame(np.append(models[:, None], votes, axis = 1))
    colnames = ['model'] + [f'rank{i}' for i in np.arange(1, len(actions)+1)]
    df_votes.columns = colnames
    df_votes.insert(1, "run", 1)
    df_votes.to_csv(join(OUTPUT_DIR, "ebola_data_votes_cases.csv"), index = False)


    # Generate votes using action labels
    votes_str = [[actions[i] for i in vote] for vote in votes]

    df_votes_str = pd.DataFrame(np.append(models[:,None], votes_str, axis = 1))
    df_votes_str.columns = colnames
    df_votes_str.insert(1, "run", 1)
    df_votes_str.to_csv(join(OUTPUT_DIR, "ebola_data_votes_str_cases.csv"), index = False)
