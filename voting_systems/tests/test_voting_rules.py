#!/usr/bin/env python3
"""
Testing script for vote-processing rules

W. Probert, 2019
"""
import numpy as np, sys
import voting_systems.core as voting

# Example 1: model 1 (the first voter/row) thought action A was the best, followed by D, then C, E, 
# and thought B was the worst action.  

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



preferences = np.array([[5, 4, 3, 2, 1], [5, 3, 2, 1, 4], [1, 2, 3, 4, 5], [1, 2, 5, 3, 4], \
    [4, 2, 3, 5, 1], [2, 1, 3, 5, 4] , [5, 3, 2, 1, 4]])


# An example that includes two votes for each candidate and at each preference
example_deadlock = np.array([[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1], [0, 2, 1], [1, 0, 2]])


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


class TestClass(object):
    """
    Class for testing vote processing rules give answers as expected.  
    """
    
    #######
    # FPP #
    #######
    
    
    def test_fpp_ties(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(winner, [1, 3])
    
    
    def test_fpp_ties_winner_index(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(winner_index, [0, 2])
    
    
    def test_fpp_ties_candidates(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(candidates, [1, 2, 3, 4, 5])
    
    
    def test_fpp_ties_tally(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1)
        np.testing.assert_array_equal(tally, [2, 1, 2, 0, 0])
    
    
    def test_fpp_char_ties(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1_char)
        np.testing.assert_array_equal(winner, ["A", "C"])
    
    
    def test_fpp_planets_ties(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example1_planets)
        np.testing.assert_array_equal(winner, ["Earth", "Mercury"])
    
    
    def test_fpp_char(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example2_char)
        np.testing.assert_equal(winner, "D")
    
    
    def test_fpp_example_deadlock_winner(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example_deadlock)
        np.testing.assert_equal(winner, [0, 1, 2])


    def test_fpp_example_deadlock_winner(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp(example_deadlock)
        np.testing.assert_equal(tally, [2, 2, 2])

    
    def test_fpp_example_trivial_ties_winner(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp([[10, 9], [9, 10]])
        np.testing.assert_equal(winner, [9, 10])


    def test_fpp_example_trivial_ties_tally(self):
        ((winner, winner_index), (candidates, tally)) = voting.fpp([[10, 9], [9, 10]])
        np.testing.assert_equal(tally, [1, 1])
        
        
    ###############
    # Borda count #
    ###############
    
    def test_borda_count_ex1_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1)
        np.testing.assert_array_equal(winner, [1, 3])
    
    
    def test_borda_count_ex1_points(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1)
        np.testing.assert_array_equal(points, [20, 15, 20,  9, 11])
        
        
    def test_borda_count_ex1_char_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_char)
        np.testing.assert_array_equal(winner, ["A", "C"])
    
    
    def test_borda_count_ex1_char_winner_index(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_char)
        np.testing.assert_array_equal(winner_index, [0, 2])
    
    
    def test_borda_count_ex1_char_planets_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_planets)
        np.testing.assert_array_equal(winner, ["Earth", "Mercury"])
    
    
    def test_borda_count_ex1_char_planets_winner_index(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1_planets)
        
        # Ordering of candidates has changed to example1_char, so in alphabetical order
        # of first 5 planets, Earth and Mercury are 1st and 4th (so 0 and 3 from Python indexing)
        np.testing.assert_array_equal(winner_index, [0, 3])
    
    
    def test_borda_count_example1a_char_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_equal(winner, "A")
    
    
    def test_borda_count_example1a_char_points(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_array_equal(points, [29, 21, 27, 12, 16])
    
    
    def test_borda_count_example1a_candidates(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_equal(candidates, ["A", "B", "C", "D", "E"])
    
    
    def test_borda_count_example1a_winner_index(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example1a_char)
        np.testing.assert_array_equal(winner_index, 0)


    def test_borda_count_example3_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example3_char)
        np.testing.assert_array_equal(winner, "B")


    def test_borda_count_example3_winner_index(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example3_char)
        np.testing.assert_array_equal(winner_index, 1)


    def test_borda_count_example_deadlock_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example_deadlock)
        np.testing.assert_array_equal(winner, [0, 1, 2])


    def test_borda_count_example_deadlock_points(self):
        (winner, winner_index), (candidates, points) = voting.borda_count(example_deadlock)
        np.testing.assert_array_equal(points, [12, 12, 12])


    def test_borda_count_example_trivial_winner(self):
        (winner, winner_index), (candidates, points) = voting.borda_count([[10, 9], [9, 10]])
        np.testing.assert_array_equal(winner, [9, 10])


    def test_borda_count_example_deadlock_points(self):
        (winner, winner_index), (candidates, points) = voting.borda_count([[10, 9], [9, 10]])
        np.testing.assert_array_equal(points, [3, 3])


    #################
    # Coombs method #
    #################
    
    def test_coombs_ex3_char_winner(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example3_char)
        np.testing.assert_equal(winner, "C")
    
    
    def test_coombs_ties(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method([[1, 2], [1, 2]])
        np.testing.assert_equal(winner, 1) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Use real example
        
        
    def test_coombs_ex1(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1)
        np.testing.assert_equal(winner, 1)
    
    
    def test_coombs_ex1_1(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1 - 1)
        np.testing.assert_equal(winner, 0)
    
    
    def test_coombs_character(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1_char)
        np.testing.assert_equal(winner, "A")
    
    
    def test_coombs_ex1_planets(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example1_planets)
        np.testing.assert_equal(winner, "Mercury")
    
    # def test_coombs_method_ties(self):
    #     winner, removed_actions = voting.coombs_method([[0, 1], [1, 0]])
    #     np.testing.assert_equal(winner, None)
    
    
    def test_coombs_method_ex5_winner(self):
        (winner, winner_index), (candidates, removed) = voting.coombs_method(example5_char)
        np.testing.assert_equal(winner, "B")
    
    
    def test_coombs_method_ex5_removed_actions(self):
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
    
    def test_alternative_vote_example1_char_winner(self):
        (winner, winner_index), (candidates, removed) = voting.alternative_vote(example1_char)
        np.testing.assert_equal(winner, "A")


    def test_alternative_vote_example1_char_removed_actions(self):
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


    def test_alternative_vote_example1_char_removed_actions(self):
        (winner, winner_index), (candidates, removed) = \
            voting.alternative_vote(example1a_char_3times)
        np.testing.assert_array_equal(removed, ["E", "D", "A"])

