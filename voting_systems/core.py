#!/usr/bin/env python3
"""
Definition of vote-processing rules, and analysis of FMD data using these rules (see bottom of script).  


W. Probert, 2019

--- To do ---
Write out examples of Coombs' method
Write out AV method.  
Write out examples of alternative vote method
Unit tests with several ties
Process data from Shouli
Votes are input in the same format, and each function outputting the winner with the same format
Deal with string inputs

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
        Array (or multi-dimensional array) of votes, rows are voters, columns are preference
        For instance [["H", "D", "A"], ["A", "D", "H"]] represents two votes, the first voter voted
        candidate H first, then candidate D, then candidate A, the second voter voted candidate A
        first, then candidate D, then candidate H.  
    
    Returns
    -------
    
    """
    
    # Check input has the same number of votes for each voter
    diff_size_to_first_vote = list(filter(lambda x: len(x) != len(votes[0]), votes))
    if len(diff_size_to_first_vote) != 0:
        raise Exception("No partial ballots allowed; some voters do not have a full ballot." +
            " Exiting.")
    
    # Check input is a numpy array or list
    if not isinstance(votes, np.ndarray):
        if isinstance(votes, list):
            votes = np.asarray(votes)
        else: 
            raise Exception("Input needs to be list or numpy array. Exiting.")
    
    # Find unique 1st preference votes
    unique_sorted, reverse = np.unique(votes[:,0], return_inverse = True)
    tally = np.bincount(reverse)
    
    # Check for ties
    maximums = np.where(tally == np.max(tally))[0]
    
    # Find the best action
    if len(maximums) == 1:
        astar_idx = np.argmax(tally)
        winner = unique_sorted[astar_idx]
    else:
        winner = unique_sorted[maximums]
    
    return(winner, tally)


def borda_count(votes):
    """
    Borda count method
    
    Arguments
    ---------
    votes: numpy array or list
        Array (or multi-dimensional array or list) of votes.  
    
    
    Returns
    -------
    tuple of (winner, points)
    
    winner : 
        the winning canidate (same object as whatever the input was)
    points : numpy array of int
        Borda score for each candidate
    
    
    Example
    -------
    
    
    sum(np.array([borda_count(df.loc[df.run == i, "duration"].values) for i in np.arange(1, 100)]))
    """
    
    candidates = np.unique(votes[0])
    
    v, n = votes.shape
    
    output = [np.unique(r, return_inverse = True) for r in votes]
    output = np.array([n - r for u, r in output])
    
    points = np.sum(output, axis = 0)
    winner = candidates[np.argmax(points)]
    
    return(winner, points)


def coombs_method(votes, verbose = False):
    """
    Arguments
    ---------
    votes: list or numpy array
        Array (or multi-dimensional array) of votes, rows are voters, columns are preference
        For instance [["H", "D", "A"], ["A", "D", "H"]] represents two votes, the first voter voted
        candidate H first, then candidate D, then candidate A, the second voter voted candidate A
        first, then candidate D, then candidate H.  
    
    
    verbose : boolean
        Should additional outputs be printed to screen.  
    
    
    Returns
    -------
    (winner, removed) : 
        winner : the winner chosen via Coombs method (None if tied).  
        removed : list of any removed candidates at different rounds of the algorithm.  
    
    Example
    -------
    votes = np.array([
        [1, 4, 3, 5, 2], 
        [1, 3, 5, 4, 2], 
        [2, 4, 1, 5, 3], 
        [3, 2, 1, 5, 4], 
        [2, 1, 3, 5, 4]]) - 1
    
    coombs_method(votes)
    
    """
    
    # Check input has the same number of votes for each voter
    diff_size_to_first_vote = list(filter(lambda x: len(x) != len(votes[0]), votes))
    if len(diff_size_to_first_vote) != 0:
        raise Exception("No partial ballots allowed; some voters do not have a full ballot." +
            " Exiting.")
    
    # Check input is a numpy array or list
    if not isinstance(votes, np.ndarray):
        if isinstance(votes, list):
            votes = np.asarray(votes)
        else: 
            raise Exception("Input needs to be list or numpy array. Exiting.")
    
    n_voters, n_actions = votes.shape
    
    # Default value for showing there's no current winner
    winner = None

    # Make a copy of current preferences
    current_votes = votes

    round_idx = 1
    
    # List of removed candidates
    removed = []
    
    # Continue removing last preference candidates until we find a winner (i.e. a majority fav)
    while winner is None:
        if verbose:
            print("Round ", round_idx)
        
        # Find the unique set of votes and tally the frequency of each type of vote
        unique_votes, idx = np.unique(current_votes, axis = 0, return_inverse = True)
        tally = np.bincount(idx)
        
        # Find number of first preferences for each candidate action
        first_preferences = unique_votes[:,0]
        
        # Tally the frequency of each candidate for first preference
        for action in np.unique(first_preferences):
            action_idx = np.where(first_preferences == action)[0]
            number_votes = np.sum(tally[action_idx])
            
            # Check if majority is reached
            if number_votes > n_voters*0.5:
                winner = action
                if verbose: 
                    print(current_votes)
                    print("Majority found, winner: ", winner)
                return(winner, removed)
        
        # If there's no winner, remove the action with the largest number of least favourite votes
        # Placeholder to determine if we've found a candidate action to remove
        loser = None
        
        # Record the level at which we're comparing votes 
        # (ie counting backwards from end of array)
        # (e.g. 1 is last preference votes, 2 is 2nd last preference votes, etc)
        preference_level = 1
        
        # Continue until we've found a candidate action to remove
        # List the last preference votes
        last_preferences = unique_votes[:, -preference_level]
        last_preference_tally = []
        for action in np.unique(last_preferences):
            action_idx = np.where(last_preferences == action)[0]
            last_preference_tally.append(np.sum(tally[action_idx]))
        
        # Find action with the most number of last 
        max_last_pref = np.max(last_preference_tally)
        last_pref = np.unique(last_preferences)[last_preference_tally == max_last_pref]
        
        if len(last_pref) == 1:
            loser = last_pref[0]
            if verbose: 
                print(current_votes)
                print("Removing ", loser)
                print("---------------")
            removed.append(loser)
        else:
            
            # If there's ties in the last preference then choose between these last candidates
            # using their 2nd to last votes, 3rd to last votes, and so on, so as to break ties. 
            while loser is None:
                # Look at next preference level to break ties
                # (for the current 'preference' level, eg last, 2nd last, 3rd last, etc)
                preference_level += 1
                next_preference_level = unique_votes[:, -preference_level]
                
                next_last_pref = [np.sum(tally[next_preference_level == last]) \
                    for last in last_pref]
                    
                next_last_preference = last_pref[next_last_pref == np.max(next_last_pref)]
                
                # If we've got a single last-preference on this preference level, then that's 
                # the candidate to remove, otherwise repeat the process on next preference level
                if len(next_last_preference) == 1:
                    loser = next_last_preference[0]
                    if verbose: 
                        print(current_votes)
                        print("Removing ", loser)
                        print("---------------")
                    removed.append(loser)
        
        # Remove the losing candidate for this round
        current_votes = np.array([list(filter(lambda x: x != loser, vote)) \
            for vote in current_votes])
        
        # Increment the round counter
        round_idx += 1


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


