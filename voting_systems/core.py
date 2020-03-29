#!/usr/bin/env python3
"""
Definition of vote-processing rules for comparing output from multiple models

Votes are input in the same format, and each function outputting the winner with the same format.  

W. Probert, 2019
"""

import copy, numpy as np


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
    (candidates, tally, (winner, winner_index))

    candidates : list
        list of all possible candidates, in the order returned using np.unique(votes[0]).  There are
        no partial ballots so each vote in 'votes' should contain all candidates.      
    tally: list
        list of votes for each candidate, in the same order as 'candidates'.  
    winner: 
        winner the winning candidate under FPP.  This is the same type as the input elements of
        'votes'.  In the case of ties a list is returned with the tied winners.  
    winner_index : int
        index of 'candidates' of the winning candidate.  In the case of ties
    
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
    tally = [np.sum(votes[:,0] == c) for c in candidates]
    
    # If counts for all candidates don't need to be returned, then can use this:
    #unique_sorted, tally = np.unique(votes[:,0], return_inverse = True)
    #tally = np.bincount(reverse)
    
    # Check for ties
    winner_index = np.where(tally == np.max(tally))[0]
    
    # Find the best action(s)
    if len(winner_index) == 1:
        winner_index = np.argmax(tally)
        winner = candidates[winner_index]
    else:
        winner = candidates[winner_index]
    
    return((winner, winner_index), (candidates, tally))


def borda_count(votes):
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
    tuple of (winner, points)
    
    winner : 
        the winning canidate (same object as whatever the input was)
    points : numpy array of int
        Borda score for each candidate
    
    
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
    (winner, removed) : 
        winner
            the winner chosen via Coombs method (None if tied).  
        removed
            list of any removed candidates at different rounds of the algorithm.  
    
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
    current_votes = votes.astype(int)

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
        
        # Record the level at which we're comparing votes 
        # (ie counting backwards from end of array)
        # (e.g. 1 is last preference votes, 2 is 2nd last preference votes, etc)
        preference_level = 1
        
        # Continue until we've found a candidate action to remove
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
                        print("Removing ", loser)
                        print("---------------")
                    removed.append(loser)
        
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
    
    Nvotes, Ncandidates = votes.shape
    candidates = np.unique(votes[0])
    
    # Default value for showing no winner has yet been determined
    winner = None

    # Make a copy of current preferences
    current_votes = votes.astype(int)

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
            while loser is None:
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
        
        # Remove the losing candidate for this round
        current_votes = np.array([np.array(vote[vote != loser]) for vote in current_votes])
        # Increment the round counter
        round_idx += 1




def alternative_vote_old(preferences):
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
    
    Nvotes, Ncandidates = preferences.shape
    
    # Calculate the percentage of votes for each preference (row) for each action (col)
    tally = [np.sum(preferences == i, axis = 0)/Nvotes for i in np.arange(1, Ncandidates + 1)]
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
        
        tally = [np.sum(second_round_preferences == i, axis = 0)/Nvotes \
            for i in np.arange(1, Ncandidates + 1)]
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
            
            tally = [np.sum(third_round_preferences == i, axis = 0)/Nvotes \
                for i in np.arange(1, Ncandidates + 1)]
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
            
                tally = [np.sum(fourth_round_preferences == i, axis = 0)/Nvotes \
                    for i in np.arange(1, Ncandidates + 1)]
                proportion_first = tally[0]
                
                if(np.any(proportion_first > 0.5)):
                    winner = np.where(proportion_first > 0.5)[0][0]
                else: 
                    winner = -1
            
    return(winner)
