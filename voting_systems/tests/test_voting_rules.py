#!/usr/bin/env python3
"""
Testing script for vote-processing rules


Votes are recorded in the following manner: 
Each row is a vote, each column is the preference/rank (note: each column is *not* a candidate).
The elements in each row give the order of preference.  


For each rule, the following has been tested:

* Test the rule winner is correct
* Test the rule winner_index is correct
* Test the rule candidate list is correct
* Test the rule 'additional output' is correct
(additional output is things like Borda counts, removed candidates etc)

* Test the rule works with integer inputs
* Test the rule works with string inputs
* Test the rule works with ties in the middle of the algorithm
* Test the rule works with no winner/ties ('deadlock')

W. Probert, 2019
"""
import numpy as np, sys
from collections import Counter

import voting_systems as voting

# Example 1: model 1 (the first voter/row) thought action A was the best, 
# followed by D, then C, E, and thought B was the worst action.  

# Example 1 as ranks
example1 = np.array([
    [1, 4, 3, 5, 2],
    [1, 3, 5, 4, 2],
    [3, 2, 1, 5, 4],
    [3, 2, 1, 5, 4],
    [2, 1, 3, 5, 4]
])

# Example 1 as using the names of the actions/candidates
example1_char = np.array([
    ["A", "D", "C", "E", "B"],
    ["A", "C", "E", "D", "B"],
    ["C", "B", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"]
])

# Same as example 1 but using planet names (so alphabetic ordering from np.unique won't be the same)
example1_planets = np.array([
    ["Mercury", "Mars", "Earth", "Jupiter", "Venus"],
    ["Mercury", "Earth", "Jupiter", "Mars", "Venus"],
    ["Earth", "Venus", "Mercury", "Jupiter", "Mars"],
    ["Earth", "Venus", "Mercury", "Jupiter", "Mars"],
    ["Venus", "Mercury", "Earth", "Jupiter", "Mars"]
])

# Example 1a (same as example 1 but with different number of candidates and votes)
# Second and final row of example1_car have been repeated to get example1a_char
# Borda count points should be A: 29; B: 21; C: 27; D: 12; E: 16
example1a_char = np.array([
    ["A", "D", "C", "E", "B"],
    ["A", "C", "E", "D", "B"],
    ["A", "C", "E", "D", "B"],
    ["C", "B", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"],
    ["B", "A", "C", "E", "D"]
])

# Example 2: model 1 (the first voter/row) thought action D was the best, followed by A, then C, E, 
# and thought B was the worst action.  

example2_char = np.array([
    ["D", "A", "C", "E", "B"],
    ["D", "C", "E", "A", "B"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"]
])


# Example 3: in this case Borda count should give the win to "B" even those "C" is the most 
# preferred candidate
example3_char = np.array([
    ["C", "B", "D", "E", "A"],
    ["C", "B", "D", "E", "A"],
    ["C", "B", "D", "E", "A"],
    ["A", "B", "E", "D", "C"],
    ["A", "B", "E", "D", "C"]
])


# Example 4: 
preferences = np.array([
    [5, 4, 3, 2, 1], 
    [5, 3, 2, 1, 4], 
    [1, 2, 3, 4, 5], 
    [2, 1, 3, 5, 4]
])

# Using this example: 
# 1) No overall majority
# 2) Action 1 should be removed first (it has the most 5's)
# 3) Any 1st preference votes for action 1 should be given to the action that that was voted second

example5_char = np.array([
    ["A", "D", "C", "E", "B"],
    ["A", "C", "E", "D", "B"],
    ["B", "D", "A", "E", "C"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"]
])



preferences = np.array([
    [5, 4, 3, 2, 1], 
    [5, 3, 2, 1, 4], 
    [1, 2, 3, 4, 5], 
    [1, 2, 5, 3, 4],
    [4, 2, 3, 5, 1], 
    [2, 1, 3, 5, 4] , 
    [5, 3, 2, 1, 4]])


# An example that includes two votes 
# for each candidate and at each preference
example_deadlock = np.array([
    [0, 1, 2], 
    [2, 1, 0], 
    [1, 2, 0], 
    [2, 0, 1], 
    [0, 2, 1], 
    [1, 0, 2]
])

example_deadlock_partial = np.array([
    [0, 1, 2, 3], 
    [2, 1, 0, 3], 
    [1, 2, 0, 3], 
    [2, 0, 1, 3], 
    [0, 2, 1, 3], 
    [1, 0, 2, 3]
])


example1a_char_3times = np.array([
    ["A", "D", "C", "E", "B"],
    ["A", "C", "E", "D", "B"],
    ["A", "C", "E", "D", "B"],
    ["C", "B", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"],
    ["B", "A", "C", "E", "D"],
    ["A", "D", "C", "E", "B"],
    ["A", "C", "E", "D", "B"],
    ["A", "C", "E", "D", "B"],
    ["C", "B", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"],
    ["B", "C", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"],
    ["B", "A", "C", "E", "D"]
])



def test_values_to_votes_1():
    
    projections = np.asarray([[6, 5, 4, 3, 2, 1]])
    
    votes = voting.values_to_votes(projections)
    
    np.testing.assert_array_equal(votes, np.asarray([[5, 4, 3, 2, 1, 0]]))


def test_values_to_votes_2():
    
    projections = np.asarray([[1.2, 2.5, 3.4, 4.3, 5.2, 6.1]])
    
    votes = voting.values_to_votes(projections)
    
    np.testing.assert_array_equal(votes, np.asarray([[0, 1, 2, 3, 4, 5]]))


def test_values_to_votes_3():
    
    projections = np.asarray([[2.5, 1.2, 4.3, 3.4, 6.1, 5.2]])
    
    votes = voting.values_to_votes(projections)
    
    np.testing.assert_array_equal(votes, np.asarray([[1, 0, 3, 2, 5, 4]]))


def test_values_to_votes_3_labels():
    
    projections = np.asarray([[2.5, 1.2, 4.3, 3.4, 6.1, 5.2]])
    
    votes = voting.values_to_votes(
        values = projections, 
        candidate_labels = ["A", "B", "C", "D", "E", "F"])
    
    np.testing.assert_array_equal(votes, np.asarray([["B", "A", "D", "C", "F", "E"]]))


def test_values_to_votes_multiple_votes():
    """
    Example with several values from examples above
    """
    
    projections = np.asarray([
            [6, 5, 4, 3, 2, 1],
            [6, 5, 4, 3, 2, 1],
            [1.2, 2.5, 3.4, 4.3, 5.2, 6.1],
            [2.5, 1.2, 4.3, 3.4, 6.1, 5.2]
            ])
    
    votes = voting.values_to_votes(projections)
    
    np.testing.assert_array_equal(votes, 
        np.asarray([
            [5, 4, 3, 2, 1, 0],
            [5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5],
            [1, 0, 3, 2, 5, 4]
        ]))


def test_values_to_votes_4():
    
    projections = np.asarray([[1, 1, 1, 1, 1, 1]])
    secondary_objective = [np.asarray([[2.5, 1.2, 4.3, 3.4, 6.1, 5.2]])] # same as #3 aboves
    
    votes = voting.values_to_votes(projections, secondary_value = secondary_objective)
    
    np.testing.assert_array_equal(votes, np.asarray([[1, 0, 3, 2, 5, 4]]))


def test_values_to_votes_4_labels():
    
    projections = np.asarray([[1, 1, 1, 1, 1, 1]])
    secondary_objective = [np.asarray([[2.5, 1.2, 4.3, 3.4, 6.1, 5.2]])] # same as #3 aboves
    
    votes = voting.values_to_votes(projections, 
        candidate_labels = ["A", "B", "C", "D", "E", "F"],
        secondary_value = secondary_objective)
    
    np.testing.assert_array_equal(votes, np.asarray([["B", "A", "D", "C", "F", "E"]]))


def test_values_to_votes_5():
    
    projections = np.asarray([[1, 2, 2, 2, 1, 1]])
    secondary_objective = [np.asarray([[2.5, 1.2, 4.3, 3.4, 6.1, 5.2]])] # same as #3 aboves
    
    votes = voting.values_to_votes(projections, secondary_value = secondary_objective)
    
    np.testing.assert_array_equal(votes, np.asarray([[0, 5, 4, 1, 3, 2]]))


def test_values_to_votes_random():
    """
    Check splitting ties using a random approach provides a random
    In 10000 votes on 6 candidates, the ties should give the same proportion of 
    preferences to each candidate if ties are split randomly.  
    """
    Nvotes = 100000
    Nactions = 4
    
    projections = np.ones((Nvotes, Nactions))
    
    votes = voting.values_to_votes(projections)
    
    # Tally the number of times each candidate was placed in a particular preference
    preference_counts = [Counter(x) for x in votes.T]
    
    # Take proportions
    preference_props = np.array([list(p.values()) for p in preference_counts])/Nvotes
    
    # Proportions should be evenly distributed (check it's the same to 2 dp)
    np.testing.assert_array_almost_equal(
        preference_props, 
        np.ones((Nactions, Nactions))*(1./Nactions), 
        decimal = 2)


class TestClass(object):
    """
    Class for testing vote processing rules give answers as expected.  
    """
    
    #######
    # FPP #
    #######
    
    def test_fpp_winner(self):
        """Test FPP winner"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example2_char)
        np.testing.assert_equal(winner, "D")
    
    def test_fpp_winner_index(self):
        """Test FPP winner index"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example2_char)
        np.testing.assert_equal(winner_index, 3)
    
    def test_fpp_candidates(self):
        """Test FPP candidate list"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example2_char)
        np.testing.assert_equal(candidates, ["A", "B", "C", "D", "E"])
    
    def test_fpp_additional_info(self):
        """Test FPP additional information (tally)"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example2_char)
        np.testing.assert_array_equal(tally, [0, 1, 1, 2, 0])
    
    
    
    def test_fpp_ties_winner(self):
        """Test FPP winner"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(winner, [1, 3])
    
    def test_fpp_ties_winner_index(self):
        """Test FPP winner index when ties occur"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(winner_index, [0, 2])
    
    def test_fpp_ties_candidates(self):
        """Test FPP candidates list when ties occur"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(candidates, [1, 2, 3, 4, 5])
    
    def test_fpp_ties_tally(self):
        """Test FPP additional output when ties occur (tally)"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(tally, [2, 1, 2, 0, 0])
    
    
    
    def test_fpp_string_ties_winner(self):
        """Test FPP winner using string inputs when ties occur"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1_planets)
        np.testing.assert_array_equal(winner, ["Earth", "Mercury"])
    
    
    
    def test_fpp_deadlock_winner(self):
        """Test FPP winner when deadlock occurs (ties between ALL candidates)"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example_deadlock)
        np.testing.assert_array_equal(winner, [0, 1, 2])

    def test_fpp_deadlock_additional_info(self):
        """Test FPP additional info (tally) when deadlock occurs (ties between ALL candidates)"""
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example_deadlock)
        np.testing.assert_array_equal(tally, [2, 2, 2])
    
        
    ###############
    # Borda count #
    ###############
    
    def test_borda_count_winner(self):
        """Test Borda count winner"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_equal(winner, "A")
    
    def test_borda_count_winner_index(self):
        """Test Borda count winner index"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_array_equal(winner_index, 0)
    
    def test_borda_count_candidates(self):
        """Test Borda count candidate list"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_equal(candidates, ["A", "B", "C", "D", "E"])
    
    def test_borda_count_additional_info(self):
        """Test Borda count additional information (Borda count score)"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_array_equal(points, [29, 21, 27, 12, 16])
    
    
    def test_borda_count_ties_winner(self):
        """
        Test Borda count winner when ties occur
        example1_char should give ties in winners with Borda count
        """
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_char)
        np.testing.assert_array_equal(winner, ["A", "C"])
    
    def test_borda_count_ties_winner_index(self):
        """Test Borda count winner index when ties occur"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_char)
        np.testing.assert_array_equal(winner_index, [0, 2])

    def test_borda_count_ties_candidate_list(self):
        """Test Borda count candidate list when ties occur"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_char)
        np.testing.assert_array_equal(candidates, ["A", "B", "C", "D", "E"])
            
    def test_borda_count_ties_additional_info(self):
        """Test Borda count additional information (Borda count score) when ties occur"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_char)
        np.testing.assert_array_equal(points, [20, 15, 20,  9, 11])
        
    
    def test_borda_count_string_ties_winner(self):
        """
        Test Borda count winner when ties occur and input candidates are strings
        example1_planets should give ties in winners with Borda count
        (alphabetical ordering of candidates is different to example1_char)
        """
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_planets)
        np.testing.assert_array_equal(winner, ["Earth", "Mercury"])
    
    def test_borda_count_string_ties_winner_index(self):
        """Test Borda count winner when ties occur and input candidates are strings"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_planets)
        # Ordering of candidates has changed to example1_char, so in alphabetical order
        # of first 5 planets, Earth and Mercury are 1st and 4th (so 0 and 3 from Python indexing)
        np.testing.assert_array_equal(winner_index, [0, 3])
        
        
    def test_borda_count_quirk_winner(self):
        """Test Borda count winner when most popular candidate in first votes does not win"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example3_char)
        np.testing.assert_array_equal(winner, "B")


    def test_borda_count_quirk_winner_index(self):
        """Test Borda count winner index when most popular candidate in first votes does not win"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example3_char)
        np.testing.assert_array_equal(winner_index, 1)



    def test_borda_count_deadlock_winner(self):
        """Test Borda Count winner when deadlock occurs (ties between ALL candidates)"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example_deadlock)
        np.testing.assert_array_equal(winner, [0, 1, 2])

    def test_borda_count_deadlock_additional_info(self):
        """Test Borda Count points when deadlock occurs (ties between ALL candidates)"""
        (winner, winner_index), (candidates, points) = voting.borda_count(example_deadlock)
        np.testing.assert_array_equal(points, [12, 12, 12])


    #################
    # Coombs method #
    #################
    
    def test_coombs_winner(self):
        """Test Coombs method winner"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example3_char)
        np.testing.assert_equal(winner, "C")
    
    def test_coombs_winner_index(self):
        """Test Coombs method winner index"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example3_char)
        np.testing.assert_equal(winner_index, 2)
    
    def test_coombs_candidates(self):
        """Test Coombs method candidate list"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example3_char)
        np.testing.assert_equal(candidates, ["A", "B", "C", "D", "E"])
    
    def test_coombs_additional_info(self):
        """Test Coombs method additional information (list of removed candidates)"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example3_char)
        np.testing.assert_equal(removed, [])
    
    
    def test_coombs_ties_winner(self):
        """Test Coombs method winner when deadlock occurs (ties between ALL candidates)"""
        (winner, winner_index), (candidates, removed) = \
                voting.coombs_method(example_deadlock, verbose = True)
        np.testing.assert_equal(winner, [0, 1, 2])

    def test_coombs_ties_winner_index(self):
        """Test Coombs method winner index when deadlock occurs (ties between ALL candidates)"""
        (winner, winner_index), (candidates, removed) = \
                voting.coombs_method(example_deadlock)
        np.testing.assert_equal(winner_index, [0, 1, 2])
    
    
    def test_coombs_ties_partial_winner(self):
        """Test Coombs method winner when deadlock occurs part-way through the algorithm
        (i.e. ties between ALL remaining candidates); candidate 3 should be removed"""
        (winner, winner_index), (candidates, removed) = \
                voting.coombs_method(example_deadlock_partial, verbose = True)
        np.testing.assert_equal(winner, [0, 1, 2])
    
    def test_coombs_ties_partial_winner_index(self):
        (winner, winner_index), (candidates, removed) = \
                voting.coombs_method(example_deadlock_partial, verbose = True)
        np.testing.assert_equal(winner, [0, 1, 2])
    
    def test_coombs_ties_partial_candidates(self):
        (winner, winner_index), (candidates, removed) = \
                voting.coombs_method(example_deadlock_partial, verbose = True)
        np.testing.assert_equal(candidates, [0, 1, 2, 3])
    
    def test_coombs_ties_partial_removed(self):
        (winner, winner_index), (candidates, removed) = \
                voting.coombs_method(example_deadlock_partial, verbose = True)
        np.testing.assert_equal(removed, [3])
    
    
    def test_coombs_string_winner(self):
        """Test Coombs method winner when using string inputs"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1_planets)
        np.testing.assert_array_equal(winner, "Mercury")
    
    def test_coombs_string_winner_index(self):
        """Test Coombs method winner index when using string inputs"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1_planets)
        np.testing.assert_array_equal(winner_index, 3)
    
    
    def test_coombs_method_ex5_winner(self):
        """Test Coombs method winner when ties occur in initial rounds"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example5_char)
        np.testing.assert_equal(winner, "B")
    
    def test_coombs_method_ex5_winner_index(self):
        """Test Coombs method winner index when ties occur in initial rounds"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example5_char)
        np.testing.assert_equal(winner_index, 1)
        
    def test_coombs_method_ex5_candidates(self):
        """Test Coombs method removed candidates when ties occur in initial rounds"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example5_char)
        np.testing.assert_array_equal(candidates, ["A", "B", "C", "D", "E"])
    
    def test_coombs_method_ex5_additional_info(self):
        """Test Coombs method removed candidates when ties occur in initial rounds"""
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example5_char)
        np.testing.assert_array_equal(removed, ["D", "E", "C"])
    
    
    def test_coombs_method_example1a_char_3times_winner(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1a_char_3times)
        np.testing.assert_equal(winner, "C")
    
    def test_coombs_method_example1a_char_3times_removed_actions(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1a_char_3times)
        np.testing.assert_array_equal(removed, ["D", "E", "A"])
    
    
    ####################
    # Alternative vote #
    ####################
    
    def test_alternative_vote_winner(self):
        """Test Alternative vote winner"""
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1_char)
        np.testing.assert_equal(winner, "A")

    def test_alternative_vote_winner_index(self):
        """Test Alternative vote winner index"""
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1_char)
        np.testing.assert_equal(winner_index, 0)
        
    def test_alternative_vote_candidates(self):
        """Test Alternative vote candidates"""
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1_char)
        np.testing.assert_array_equal(candidates, ["A", "B", "C", "D", "E"])
        
    def test_alternative_vote_additional_info(self):
        """Test Alternative vote removed list"""
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1_char)
        np.testing.assert_array_equal(removed, ["E", "D", "B"])
    
    
    
    def test_alternative_vote_example1a_char_winner(self):
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1a_char)
        np.testing.assert_equal(winner, "A")

    def test_alternative_vote_example1a_char_removed_actions(self):
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1a_char)
        np.testing.assert_array_equal(removed, ["E", "D", "B"])
    
    
    
    def test_alternative_vote_example1a_char_3times_winner(self):
        (winner, winner_index), (candidates, removed) = \
            voting.alternative_vote(example1a_char_3times)
        np.testing.assert_equal(winner, "C")
        
    def test_alternative_vote_example1a_char_removed_actions(self):
        (winner, winner_index), (candidates, removed) = \
            voting.alternative_vote(example1a_char_3times)
        np.testing.assert_array_equal(removed, ["E", "D", "A"])
