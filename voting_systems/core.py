#!/usr/bin/env python3
"""
Definition of vote-processing rules for comparing output from multiple models

Votes are input in the same format, and each function outputting the winner with the same format.  

W. Probert, 2019
"""

import copy, numpy as np, pandas as pd

def values_to_votes(values, **kwargs):
    """
    Convert model output across different actions into rankings of actions
    
    Arguments
    ---------
    
    values: 2D np.array
        Model projections under different actions.  Rows are models (or model replicates), columns
        are possible actions.  Assumes all models have output for all actions (i.e. no partial
        ballots/votes are possible).  Ranking is performed in ascending order (see np.argsort).  
    
    kwargs: optional
        secondary_value : list with each element having same shape as 'values'
        Secondary values upon which to split ties.  Passed in order of preference.  Multiple arrays
        can be passed.  Sorting is performed in ascending order using np.lexsort.  
    
        candidate_labels : list
        List of candidate labels (if not input then indices of candidates are returned, indices
        found by calling np.unique() on the 'values' argument).
    
    Returns
    -------
    
    votes: np.array (same shape as 'values')
        Rankings of actions within each row of 'values'.  Rows are models (in same order as 'votes),
        columns are rank preference.  For instance, votes = np.array([[2, 0, 1], [1, 0, 2]]) would
        mean there were two votes the first vote had the 3rd candidate action as best, followed by
        the 1st, followed by the 2nd, the second vote ranked the 2nd candidate first, followed by
        the 1st, followed by the 3rd.  
    """
    
    # Check type of argument (pd.DataFrame or np.array)
    # if isinstance(values, np.ndarray):
    #     pass
    #
    # if isinstance(values, pd.DataFrame):
    #     pass
    
    N_votes, N_candidates = values.shape
    
    values_to_sort = [values]
    
    if "secondary_value" in kwargs:
        # Concatentate list 'values_to_sort' with the 'secondary_value' keywork arg
        values_to_sort += kwargs["secondary_value"]
    
    # Generate a random matrix the same shape as `values` on which to split ties
    tie_breaker = np.random.uniform(0, 1, values.shape)
    values_to_sort.append(tie_breaker)
    
    # Reverse the order of the list (lexsort orders on the right-most entry first)
    values_to_sort = values_to_sort[::-1]
    
    # Apply np.lexsort on list of values
    votes = []
    for value in zip(*values_to_sort):
        votes.append(np.lexsort(value))
    
    # If 'candidate_labels' are passed, provide votes using such labels (instead of indices)
    if "candidate_labels" in kwargs:
        candidate_labels = kwargs["candidate_labels"]
        candidate_labels = np.asarray(candidate_labels)
        votes = [candidate_labels[np.asarray(v)] for v in votes]
    
    return np.array(votes)


def fpp(votes, verbose = False):
    """
    First past the post voting rule
    
    Arguments
    ---------
    votes: numpy array
        Array (or multi-dimensional array) of votes, rows are voters, columns are preference.  
        For instance [["H", "D", "A"], ["A", "D", "H"]] represents two votes, the first voter voted
        candidate H first, then candidate D, then candidate A, the second voter voted candidate A
        first, then candidate D, then candidate H.  
        
    Returns
    -------
    (winner, winner_index), (candidates, tally)

    winner: 
        winner the winning candidate.  This is the same type as the input elements of
        'votes'.  In the case of ties a list is returned with the tied winners.  
    winner_index : int
        index of 'candidates' of the winning candidate.  In the case of ties
    candidates : list
        list of all possible candidates, in the order returned using np.unique(votes[0]).  There are
        no partial ballots so each vote in 'votes' should contain all candidates.      
    tally: list
        list of votes for each candidate, in the same order as 'candidates'.  
    
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
    
    # List all candidates (there are no partial ballots so each vote contains all candidates)
    candidates = np.unique(votes[0])
    
    # Find unique 1st preference votes from list of all candidates
    tally = [np.sum(votes[:, 0] == c) for c in candidates]
    
    # Check for ties
    winner_index = np.where(tally == np.max(tally))[0]
    
    # Find the best action(s)
    if len(winner_index) == 1:
        winner_index = np.argmax(tally)
        winner = candidates[winner_index]
    else:
        winner = candidates[winner_index]
    
    return((winner, winner_index), (candidates, tally))


def borda_count(votes, verbose = False):
    """
    Borda count method
    
    Arguments
    ---------
    votes: numpy array
        Array (or multi-dimensional array) of votes, rows are voters, columns are preference.  
        For instance [["H", "D", "A"], ["A", "D", "H"]] represents two votes, the first voter voted
        candidate H first, then candidate D, then candidate A, the second voter voted candidate A
        first, then candidate D, then candidate H.  
    
    
    Returns
    -------
    (winner, winner_index), (candidates, points_per_candidate)
    
    winner: 
        winner the winning candidate.  This is the same type as the input elements of
        'votes'.  In the case of ties a list is returned with the tied winners.  
    winner_index : int
        index of 'candidates' of the winning candidate.  In the case of ties
    candidates : list
        list of all possible candidates, in the order returned using np.unique(votes[0]).  There are
        no partial ballots so each vote in 'votes' should contain all candidates. 
    points_per_candidate : numpy array of int
        Score under the Borda count system for each candidate
    
    Example
    -------
    
    
    sum(np.array([borda_count(df.loc[df.run == i, "duration"].values) for i in np.arange(1, 100)]))
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
    
    # List all candidates (there are no partial ballots so each vote contains all candidates)
    candidates = np.unique(votes[0])
    
    Nvotes, Ncandidates = votes.shape
    
    # Return rankings of each candidate within each vote (in the order of 'candidates')
    rankings_per_vote = [sorted(range(Ncandidates), key = lambda k: vote[k]) for vote in votes]
    rankings_per_vote = np.asarray(rankings_per_vote)
    
    # Adjust rankings to points within each vote
    points_per_vote = np.array([Ncandidates - ranks for ranks in rankings_per_vote])
    
    # Sum points across candidates
    points_per_candidate = np.sum(points_per_vote, axis = 0)
    
    # Check for ties
    winner_index = np.where(points_per_candidate == np.max(points_per_candidate))[0]
    
    # Find the best action(s)
    if len(winner_index) == 1:
        winner_index = np.argmax(points_per_candidate)
        winner = candidates[winner_index]
    else:
        winner = candidates[winner_index]
    
    return((winner, winner_index), (candidates, points_per_candidate))


def coombs_method(votes, verbose = False):
    """
    Coombs method vote-processing rule
    
    Coombs Method finds the number of first preferences for each candidate.  If there's a majority
    in first preferences, that candidate is the winner.  If there's no winner, keep removing 
    candidates with the largest number of least favourite votes until and absolute majority is 
    found.
    
    
    Arguments
    ---------
    votes: numpy array
        Array (or multi-dimensional array) of votes, rows are voters, columns are preference.  
        For instance [["H", "D", "A"], ["A", "D", "H"]] represents two votes, the first voter voted
        candidate H first, then candidate D, then candidate A, the second voter voted candidate A
        first, then candidate D, then candidate H.  
    
    verbose : boolean
        Should additional outputs be printed to screen.  
    
    Returns
    -------
    (winner, winner_index), (candidates, removed)
    
    winner: 
        winner the winning candidate.  This is the same type as the input elements of
        'votes'.  In the case of ties a list is returned with the tied winners.  
    
    winner_index : int
        index of 'candidates' of the winning candidate.  In the case of ties
    
    candidates : list
        list of all possible candidates, in the order returned using np.unique(votes[0]).  There are
        no partial ballots so each vote in 'votes' should contain all candidates. 
    
    removed : list
        list of removed candidates at different rounds of the algorithm (if any).  
    
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
    
    # List all candidates (there are no partial ballots so each vote contains all candidates)
    candidates = np.unique(votes[0])
    
    Nvotes, Ncandidates = votes.shape
    
    # Default value for showing no winner has yet been determined
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
            action_idx = np.where(first_preferences == action)
            number_votes = np.sum(tally[action_idx])
            
            # Check if majority is reached
            if number_votes > Nvotes*0.5:
                winner = action
                winner_index = np.where(candidates == winner)[0]
                if verbose: 
                    print("Majority found, winner: ", winner)
                return((winner, winner_index), (candidates, removed))
        
        # If there's no winner, remove the action with the largest number of least favourite votes
        # Placeholder to determine if we've found a candidate action to remove
        loser = None
        
        # Record the level at which we're comparing votes (ie counting backwards from end of array)
        # (e.g. 1 is last preference votes, 2 is 2nd last preference votes, etc)
        preference_level = 1
        
        # List the last preference votes
        last_preferences = unique_votes[:, -preference_level]
        last_preference_tally = []
        for action in np.unique(last_preferences):
            action_idx = np.where(last_preferences == action)
            last_preference_tally.append(np.sum(tally[action_idx]))
        
        # Find action with the most number of last 
        max_last_pref = np.max(last_preference_tally)
        last_pref = np.unique(last_preferences)[last_preference_tally == max_last_pref]
        
        if len(last_pref) == 1:
            loser = last_pref[0]
            if verbose: 
                print("Removing ", loser)
                print("---------------")
            removed.append(loser)
        else:
            
            # If there's ties in the last preference then choose between these last candidates
            # using their 2nd to last votes, 3rd to last votes, and so on, so as to break ties. 
            while (loser is None) and (preference_level < (unique_votes.shape[1]-1)):
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
                        print("Removing ", loser)
                        print("---------------")
                    removed.append(loser)
                    break
            else: 
                if verbose: 
                    print("Ties.  Majority not found.")
                    print("---------------")
                winner = np.setdiff1d(candidates, removed)
                winner_index = np.array([np.where(c == candidates)[0][0] for c in winner])
                return((winner, winner_index), (candidates, removed))
        
        # Remove the losing candidate for this round
        current_votes = np.array([list(filter(lambda x: x != loser, vote)) \
                                                            for vote in current_votes])
        
        # Increment the round counter
        round_idx += 1



def alternative_vote(votes, verbose = False):
    """
    Alternative vote method
    
    Find the proportion of votes for each action.  Remove the least liked candidate at each round 
    until a majority preference is found.  
    
    
    Arguments
    ---------
    votes: numpy array
        Array (or multi-dimensional array) of votes, rows are voters, columns are preference.  
        For instance [["H", "D", "A"], ["A", "D", "H"]] represents two votes, the first voter voted
        candidate H first, then candidate D, then candidate A, the second voter voted candidate A
        first, then candidate D, then candidate H.  
    
    verbose : boolean
        Should additional outputs be printed to screen.  
    
    Returns
    -------
    (winner, winner_index), (candidates, removed)
    
    winner: 
        winner the winning candidate.  This is the same type as the input elements of
        'votes'.  In the case of ties a list is returned with the tied winners.  
    
    winner_index : int
        index of 'candidates' of the winning candidate.  In the case of ties
    
    candidates : list
        list of all possible candidates, in the order returned using np.unique(votes[0]).  There are
        no partial ballots so each vote in 'votes' should contain all candidates. 
    
    removed : list
        list of removed candidates at different rounds of the algorithm (if any).
    
    Example
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
    
    Nvotes, Ncandidates = votes.shape
    candidates = np.unique(votes[0])
    
    # Default value for showing no winner has yet been determined
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
        
        remaining_candidates = np.unique(current_votes[0])
        
        # Find the unique set of votes and tally the frequency of each type of vote
        unique_votes, idx = np.unique(current_votes, axis = 0, return_inverse = True)
        tally = np.bincount(idx)
        
        # Find number of first preferences for each candidate action
        first_preferences = unique_votes[:,0]
        
        # Tally the frequency of each candidate as a first preference
        for action in remaining_candidates:
            action_idx = np.where(first_preferences == action)
            number_votes = np.sum(tally[action_idx])
            
            # Check if majority is reached
            if number_votes > Nvotes*0.5:
                winner = action
                winner_index = np.where(candidates == winner)[0]
                if verbose: 
                    print("Majority found, winner: ", winner)
                return((winner, winner_index), (candidates, removed))
        
        # If there's no winner, remove the action with the smallest number of first preference votes
        # Placeholder to determine if we've found a candidate action to remove
        loser = None
        
        # Record the level at which we're comparing votes 
        # (ie counting backwards from end of array)
        # (e.g. 1 is last preference votes, 2 is 2nd last preference votes, etc)
        preference_level = 0
        
        # Continue until we've found a candidate action to remove
        # List the last preference votes
        first_preferences = unique_votes[:, preference_level]
        first_preference_tally = []
        for action in remaining_candidates:
            action_idx = np.where(first_preferences == action)
            first_preference_tally.append(np.sum(tally[action_idx]))
        
        first_preference_tally = np.asarray(first_preference_tally)
        
        # Find action with the least number of first preferences
        min_first_pref = np.where(first_preference_tally == np.min(first_preference_tally))
        least_first_pref = remaining_candidates[min_first_pref]
        
        if len(least_first_pref) == 1:
            loser = least_first_pref[0]
            if verbose: 
                print("Removing ", loser)
                print("---------------")
            removed.append(loser)
        else:
            
            # If there's ties in the least first preference then choose between these last
            # candidates using their 2nd votes, 3rd votes, and so on, so as to break ties. 
            while (loser is None) and (preference_level < (unique_votes.shape[1]-1)):
                # Look at next preference level to break ties
                # (for the current 'preference' level, eg 2nd, 3rd, etc)
                preference_level += 1
                
                next_preference_level = unique_votes[:, preference_level]
                
                next_least_pref = [np.sum(tally[next_preference_level == least_first]) \
                    for least_first in least_first_pref]
                
                next_last_preference = least_first_pref[next_least_pref == np.min(next_least_pref)]
                
                # If we've got a single last-preference on this preference level, then that's 
                # the candidate to remove, otherwise repeat the process on next preference level
                if len(next_last_preference) == 1:
                    loser = next_last_preference[0]
                    if verbose: 
                        print("Removing ", loser)
                        print("---------------")
                    removed.append(loser)
                    break
            else: 
                if verbose: 
                    print("Ties.  Majority not found.")
                    print("---------------")
                winner = np.setdiff1d(candidates, removed)
                winner_index = np.array([np.where(c == candidates)[0][0] for c in winner])
                return((winner, winner_index), (candidates, removed))
        
        # Remove the losing candidate for this round
        current_votes = np.array([np.array(vote[vote != loser]) for vote in current_votes])
        # Increment the round counter
        round_idx += 1

