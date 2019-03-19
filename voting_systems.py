#!/usr/bin/env python3
"""
Definition of vote-processing rules, and analysis of FMD data using these rules (see bottom of script).  


W. Probert, 2019

--- To do ---
Write out AV method.  
Write out examples of Coombs' method
Write out examples of alternative vote method
What is Hare method?  
Unit tests with several ties
Process data from Shouli

Use same "vote processing rule" terminology throughout the document.  
"""

import copy
import pandas as pd
import numpy as np
from os.path import join

def fpp(votes):
    """
    First pass the post
    
    Arguments
    ---------
    votes: numpy array
        Array (or multi-dimensional array) of votes
    """
    unique_sorted, reverse = np.unique(votes, return_inverse = True)
    
    order = reverse + 1
    
    #order = np.argsort(votes)
    
    # Find the best action
    astar = np.argmin(votes)
    
    # Deal with ties (?)
    #if( np.sum(votes == np.min) ):
    
    return(astar, order)


def borda_count(votes):
    """
    Borda count method
    
    Arguments
    ---------
    votes: numpy array
        Array (or multi-dimensional array) of votes
    
    
    Returns
    -------
    tuple of (winner, points)
    
    winner : int
        index of the winning canidate
    points : numpy array of int
        Borda score for each candidate
    
    
    Example
    -------
    
    
    sum(np.array([borda_count(df.loc[df.run == i, "duration"].values) for i in np.arange(1, 100)]))
    """
    
    v, n = votes.shape
    
    output = [np.unique(r, return_inverse = True) for r in votes]
    output = np.array([n - r for u, r in output])
    
    points = np.sum(output, axis = 0)
    winner = np.argmax(points)
    
    return(winner, points)


def coombs(preferences):
    """
    Coombs method
    
    Find the proportion of votes for each action.  Remove the most disliked candidate at each 
    time point.  
    
    There can be no ties in input votes.  
    
    unique_sorted, reverse = np.unique(v, return_inverse = True)
    ordering = reverse + 1
    
    Arguments
    ---------
    preferences: numpy array
        Array (or multi-dimensional array) of votes
    
    
    Returns
    -------
    winner : int
        index of the winning canidate    
    
    Example
    -------
    
    """
    
    n_voters, n_action = preferences.shape
    
    # Calculate the percentage of votes for each preference (row) for each action (col)
    tally = [np.sum(preferences == i, axis = 0)/n_voters for i in np.arange(1, n_action + 1)]
    tally = np.array(tally)
    
    print("Running Coombs method")
    
    # Calculate the proportion of 'first' preferences each candidate got
    proportion_first = tally[0]
    
    # If there is an outright majority, return that winner
    if(np.any(proportion_first > 0.5)):
        winner = np.where(proportion_first > 0.5)[0][0]
    else:
        # First rejected action is that with the least number of favourite picks
        proportion_last = tally[-1]
        loser = np.argmax(proportion_last)
        print("Removed: ", loser)
        # What if there are ties in this?  
        
        # Votes for this action are given to the second choice
        # this could be a tuple of indices fo need to make this a general rule
        first_pref_for_loser = (preferences[:,loser] == 1)
        indices_first_pref_for_loser = np.where(first_pref_for_loser)[0]
        
        # Find complete indices of second place
        seconds = [(r, np.where(preferences[r] == 2)[0][0]) for r in indices_first_pref_for_loser]
        
        # Assign points to second choice for the action that was eliminated
        second_round_preferences = copy.copy(preferences)
        second_round_preferences[:, loser] = 0
        for coords in seconds:
            r, c = coords
            second_round_preferences[r, c] = 1
        
        tally = [np.sum(second_round_preferences == i, axis = 0)/n_voters \
            for i in np.arange(1, n_action + 1)]
        proportion_first = tally[0]
        
        if(np.any(proportion_first > 0.5)):
            winner = np.where(proportion_first > 0.5)[0][0]
        else:
            proportion_last = tally[-1]
            loser = np.argmax(proportion_last)
            print("Removed: ", loser)
            
            first_pref_for_loser = (second_round_preferences[:,loser] == 1)
            indices_first_pref_for_loser = np.where(first_pref_for_loser)[0]
            
            next_best = [second_round_preferences[r] for r in indices_first_pref_for_loser]
            next_best = [r[r != 0] for r in next_best]
            next_best = [r[r != 1] for r in next_best]
            next_best = [np.min(r) for r in next_best]
            
            # Find complete indices of second place
            seconds = [(r, np.where(second_round_preferences[r] == i)[0][0]) \
                for i, r in zip(next_best, indices_first_pref_for_loser)]
            
            # Assign points to second choice for the action that was eliminated
            third_round_preferences = copy.copy(second_round_preferences)
            third_round_preferences[:, loser] = 0
            for coords in seconds:
                r, c = coords
                third_round_preferences[r, c] = 1
            
            tally = [np.sum(third_round_preferences == i, axis = 0)/n_voters \
                for i in np.arange(1, n_action + 1)]
            proportion_first = tally[0]
            
            if(np.any(proportion_first > 0.5)):
                winner = np.where(proportion_first > 0.5)[0][0]
            else:
                
                proportion_last = tally[-1]
                loser = np.argmax(proportion_last)
                print("Removed: ", loser)
                
                first_pref_for_loser = (third_round_preferences[:,loser] == 1)
                indices_first_pref_for_loser = np.where(first_pref_for_loser)[0]
            
                # Find complete indices of second place (or it might be third place)
                next_best = [third_round_preferences[r] for r in indices_first_pref_for_loser]
                next_best = [r[r != 0] for r in next_best]
                next_best = [r[r != 1] for r in next_best]
                next_best = [np.min(r) for r in next_best]
                
                seconds = [(r, np.where(third_round_preferences[r] == i)[0][0]) \
                    for i, r in zip(next_best, indices_first_pref_for_loser)]
            
                # Assign points to second choice for the action that was eliminated
                fourth_round_preferences = copy.copy(third_round_preferences)
                fourth_round_preferences[:, loser] = 0
                for coords in seconds:
                    r, c = coords
                    fourth_round_preferences[r, c] = 1
            
                tally = [np.sum(fourth_round_preferences == i, axis = 0)/n_voters \
                    for i in np.arange(1, n_action + 1)]
                proportion_first = tally[0]
                
                if(np.any(proportion_first > 0.5)):
                    winner = np.where(proportion_first > 0.5)[0][0]
                else: 
                    winner = -1
            
    return(winner)



def alternative_vote(preferences):
    """
    Alternative vote method
    
    Find the proportion of votes for each action.  Remove the least liked candidate at each round 
    until a majority preference is found.  
    
    
    Arguments
    ---------
    preferences: numpy array
        Array (or multi-dimensional array) of votes
        There can be no ties in input votes.  
    
    Returns
    -------
    winner : int
        index of the winning canidate    
    
    Example
    -------
    """
    
    n_voters, n_action = preferences.shape
    
    # Calculate the percentage of votes for each preference (row) for each action (col)
    tally = [np.sum(preferences == i, axis = 0)/n_voters for i in np.arange(1, n_action + 1)]
    tally = np.array(tally)
    
    
    # Calculate the proportion of 'first' preferences each candidate got
    proportion_first = tally[0]
    print("Tally :", tally[0])
    # If there is an outright majority, return that winner
    if(np.any(proportion_first > 0.5)):
        winner = np.where(proportion_first > 0.5)[0][0]
    else:
        # First rejected action is that with the least number of favourite picks
        loser = np.argmin(proportion_first)
        # What if there are ties in this?  
        print("Removed", loser)
        # Votes for this action are given to the second choice
        # this could be a tuple of indices fo need to make this a general rule
        first_pref_for_loser = (preferences[:,loser] == np.min(preferences[:,loser]))
        indices_first_pref_for_loser = np.where(first_pref_for_loser)[0]
        
        # Find complete indices of second place
        seconds = [(r, np.where(preferences[r] == 2)[0][0]) for r in indices_first_pref_for_loser]
        
        # Assign points to second choice for the action that was eliminated
        second_round_preferences = copy.copy(preferences)
        second_round_preferences[:, loser] = 0
        for coords in seconds:
            r, c = coords
            second_round_preferences[r, c] = 1
        
        tally = [np.sum(second_round_preferences == i, axis = 0)/n_voters \
            for i in np.arange(1, n_action + 1)]
        proportion_first = tally[0]
        print("Tally :", tally[0])
        if(np.any(proportion_first > 0.5)):
            winner = np.where(proportion_first > 0.5)[0][0]
        else:
            
            # Remove the zeros ... 
            nonzero = np.where(proportion_first != 0)[0]
            print("Nonzero: ", nonzero)
            loser = np.argmin(proportion_first[nonzero])
            print(loser)
            loser = nonzero[loser]
            print("Removed", loser)
            first_pref_for_loser = (second_round_preferences[:,loser] == 1)
            indices_first_pref_for_loser = np.where(first_pref_for_loser)[0]
            
            next_best = [second_round_preferences[r] for r in indices_first_pref_for_loser]
            next_best = [r[r != 0] for r in next_best]
            next_best = [r[r != 1] for r in next_best]
            next_best = [np.min(r) for r in next_best]
            
            # Find complete indices of second place
            seconds = [(r, np.where(second_round_preferences[r] == i)[0][0]) \
                for i, r in zip(next_best, indices_first_pref_for_loser)]
            
            # Assign points to second choice for the action that was eliminated
            third_round_preferences = copy.copy(second_round_preferences)
            third_round_preferences[:, loser] = 0
            for coords in seconds:
                r, c = coords
                third_round_preferences[r, c] = 1
            
            tally = [np.sum(third_round_preferences == i, axis = 0)/n_voters \
                for i in np.arange(1, n_action + 1)]
            proportion_first = tally[0]
            print("Tally :", tally[0])
            if(np.any(proportion_first > 0.5)):
                winner = np.where(proportion_first > 0.5)[0][0]
            else:
                nonzero = np.where(proportion_first != 0)[0]
                print("Nonzero: ", nonzero)
                loser = np.argmin(proportion_first[nonzero])
                print(loser)
                loser = nonzero[loser]
                print("Removed", loser)
                first_pref_for_loser = (third_round_preferences[:,loser] == 1)
                indices_first_pref_for_loser = np.where(first_pref_for_loser)[0]
            
                # Find complete indices of second place (or it might be third place)
                next_best = [third_round_preferences[r] for r in indices_first_pref_for_loser]
                next_best = [r[r != 0] for r in next_best]
                next_best = [r[r != 1] for r in next_best]
                next_best = [np.min(r) for r in next_best]
                
                seconds = [(r, np.where(third_round_preferences[r] == i)[0][0]) \
                    for i, r in zip(next_best, indices_first_pref_for_loser)]
            
                # Assign points to second choice for the action that was eliminated
                fourth_round_preferences = copy.copy(third_round_preferences)
                fourth_round_preferences[:, loser] = 0
                for coords in seconds:
                    r, c = coords
                    fourth_round_preferences[r, c] = 1
            
                tally = [np.sum(fourth_round_preferences == i, axis = 0)/n_voters \
                    for i in np.arange(1, n_action + 1)]
                proportion_first = tally[0]
                
                if(np.any(proportion_first > 0.5)):
                    winner = np.where(proportion_first > 0.5)[0][0]
                else: 
                    winner = -1
            
    return(winner)


if __name__ == "__main__":
    # If this script is called as a standalone script then the following analysis is run.  
    
    # 'model_b.csv' ignored because it can only do 4 control measures
    datasets = ['model_a.csv', 'model_c.csv', 'model_d.csv', 'model_e.csv']
    
    results = []; all_durations = []; all_cattle_culled = []
    for data in datasets:
        df = pd.read_csv(join("data", data))
        
        # Pull out data on duration
        durations = df.pivot(index = 'run', columns = 'control', values = 'duration')
        durations = durations.reset_index()
        durations = durations[['ip', 'ipdc', 'rc', 'v03', 'v10']]
        
        all_durations.append(durations)
        
        # Pull out data on cattle culled
        cattle_culled = df.pivot(index = 'run', columns = 'control', values = 'cattle_culled')
        cattle_culled = cattle_culled.reset_index()
        cattle_culled = cattle_culled[['ip', 'ipdc', 'rc', 'v03', 'v10']]
        
        all_cattle_culled.append(cattle_culled)
    
    # Combine results into one array
    durations = pd.concat(all_durations); durations = np.asarray(durations)
    cattle_culled = pd.concat(all_cattle_culled); cattle_culled = np.asarray(cattle_culled)
    
    n_votes, n_actions = durations.shape
    
    # Find the percentage of runs that have ties in minimum number of cattle culled
    min_cattle_culled = [np.sum(np.min(c) == c) for c in cattle_culled]
    tiesc = np.mean(np.asarray(min_cattle_culled) > 1)*100
    print("Proportion of ties on cattle culled: ", np.around(tiesc, 2), "%")
    
    # Count the number of ties on duration after also tying on cattle culled
    min_duration = [np.sum(np.min(c) == c) for c in durations[np.asarray(min_cattle_culled) > 1]]
    tiesd = np.mean(np.asarray(min_duration) > 1)*100
    print("Proportion of ties on duration after tying on cattle culled: ", np.around(tiesd, 2), "%")
    
    
    # Sort by duration (descending), then by cattle culled (descending), then by random number
    np.random.seed(100)
    indices = [np.lexsort((r, c, d)) \
        for d, c, r in zip(durations, cattle_culled, np.random.rand(n_votes, n_actions))]
    
    # Sort by cattle culled (descending), then duration (descending), then by random number
    np.random.seed(100)
    indices = [np.lexsort((r, d, c)) \
        for d, c, r in zip(durations, cattle_culled, np.random.rand(n_votes, n_actions))]
    
    # 'indices' returns the indices of the array needed to order on
    orderings = np.asarray([np.argsort(ind) + 1 for ind in indices])
    print(orderings)
    print(np.asarray(orderings).shape)
    
    print(np.unique(np.array(orderings)[:,0], return_counts = True)[1]/399)
    print(np.unique(np.array(orderings)[:,1], return_counts = True)[1]/399)
    print(np.unique(np.array(orderings)[:,2], return_counts = True)[1]/399)
    print(np.unique(np.array(orderings)[:,3], return_counts = True)[1]/399)
    print(np.unique(np.array(orderings)[:,4], return_counts = True)[1]/399)
    
    print("\n\n\n\n\n")
    print("1. Objectives Matter data")
    
    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(orderings))
    
    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(orderings))
    
    # Borda count
    print("---Borda count---")
    winner, points = borda_count(orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)
    
    # Majority rule
    print("---First-past-the-post---")
    tally = np.mean(orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))
    
    quit()
    
    print("\n\n\n\n\n")
    print("2a. Additional model with all 1's for first action")
    print("(randomly allocating other preferences)")
    N = 100
    new_model_prefs = [[1] + list(np.random.permutation([2, 3, 4, 5])) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))

    ####################

    print("\n\n\n\n\n")
    print("2b. Additional model with all 1's for second action")
    print("(randomly allocating other preferences)")
    N = 100
    new_model_prefs = [np.random.permutation([2, 3, 4, 5]) for i in range(N)]
    new_model_prefs = [[x[0], 1, x[1], x[2], x[3]] for x in new_model_prefs]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))
    
    ####################

    print("\n\n\n\n\n")
    print("2c. Additional model with all 1's for third action")
    print("(randomly allocating other preferences)")
    N = 100
    new_model_prefs = [np.random.permutation([2, 3, 4, 5]) for i in range(N)]
    new_model_prefs = [[x[0], x[1], 1, x[2], x[3]] for x in new_model_prefs]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))
    
    ####################

    print("\n\n\n\n\n")
    print("2d. Additional model with all 1's for fourth action")
    print("(randomly allocating other preferences)")
    N = 100
    new_model_prefs = [np.random.permutation([2, 3, 4, 5]) for i in range(N)]
    new_model_prefs = [[x[0], x[1], x[2], 1, x[3]] for x in new_model_prefs]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))
    
    ####################

    print("\n\n\n\n\n")
    print("2e. Additional model with all 1's for fifth action")
    print("(randomly allocating other preferences)")
    N = 100
    new_model_prefs = [np.random.permutation([2, 3, 4, 5]) for i in range(N)]
    new_model_prefs = [[x[0], x[1], x[2], x[3], 1] for x in new_model_prefs]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))
    
    ####################

    print("\n\n\n\n\n")
    print("2f. Two additional models with all 1's for first action")
    print("(randomly allocating other preferences)")
    N = 200
    new_model_prefs = [[1] + list(np.random.permutation([2, 3, 4, 5])) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))


    print("\n\n\n\n\n")
    print("3a. One new model randomly allocated preferences")
    N = 100
    new_model_prefs = [np.random.permutation([1, 2, 3, 4, 5]) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))
    
    
    print("\n\n\n\n\n")
    print("3b. Two models with randomly allocated preferences")
    # Increase number of models with random allocation ...
    # For loop

    N = 200
    new_model_prefs = [np.random.permutation([1, 2, 3, 4, 5]) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))

    print("\n\n\n\n\n")
    print("3c. Three models with randomly allocated preferences")
    # Increase number of models with random allocation ...
    # For loop!

    N = 300
    new_model_prefs = [np.random.permutation([1, 2, 3, 4, 5]) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))

    print("\n\n\n\n\n")
    print("3d. Twenty three models with randomly allocated preferences")
    # Increase number of models with random allocation ...
    # For loop!

    N = 2300
    new_model_prefs = [np.random.permutation([1, 2, 3, 4, 5]) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))

    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))


    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))


    print("\n\n\n\n\n")
    print("3e. Thirty models with randomly allocated preferences")
    # Increase number of models with random allocation ...
    # For loop!

    N = 30000
    new_model_prefs = [np.random.permutation([1, 2, 3, 4, 5]) for i in range(N)]
    new_orderings = np.vstack((orderings, new_model_prefs))

    # Alternative vote
    print("---Alternative vote---")
    print("Winner under Alternative Vote: ", alternative_vote(new_orderings))
    
    # Coombs' method
    print("---Coombs method---")
    print("Winner under Coombs method: ", coombs(new_orderings))

    # Borda count
    print("---Borda count---")
    winner, points = borda_count(new_orderings)
    print("Points under Borda count: ", np.around(points, 2))
    print("Winner in Borda count: ", winner)

    print("---First-past-the-post---")
    tally = np.mean(new_orderings == 1, axis = 0)
    print("Proportion of first preferences: ", np.around(tally, 2))
    print("Winner under FPP: ", np.argmax(tally))

