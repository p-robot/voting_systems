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


# Example 3: 
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


preferences = np.array([[5, 4, 3, 2, 1], [5, 3, 2, 1, 4], [1, 2, 3, 4, 5], [1, 2, 5, 3, 4], \
    [4, 2, 3, 5, 1], [2, 1, 3, 5, 4] , [5, 3, 2, 1, 4]])



class TestClass(object):
    """
    Set up test class for checking
    """
    
    #######
    # FPP #
    #######
    
    def test_fpp_ties(self):
        (candidates, tally, (winner, winner_index)) = voting.fpp(example1)
        np.testing.assert_array_equal(winner, [1, 3])
    
    def test_fpp_ties_winner_index(self):
        (candidates, tally, (winner, winner_index)) = voting.fpp(example1)
        np.testing.assert_array_equal(winner_index, [0, 2])
    
    def test_fpp_ties_candidates(self):
        (candidates, tally, (winner, winner_index)) = voting.fpp(example1)
        np.testing.assert_array_equal(candidates, [1, 2, 3, 4, 5])
    
    def test_fpp_ties_tally(self):
        (candidates, tally, (winner, winner_index)) = voting.fpp(example1)
        np.testing.assert_array_equal(tally, [2, 1, 2, 0, 0])
    
    def test_fpp_char_ties(self):
        (candidates, tally, (winner, winner_index)) = voting.fpp(example1_char)
        np.testing.assert_array_equal(winner, ["A", "C"])
    
    def test_fpp_char(self):
        (candidates, tally, (winner, winner_index)) = voting.fpp(example2_char)
        np.testing.assert_equal(winner, "D")
    
    
    ###############
    # Borda count #
    ###############
    
    def test_borda_count(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1)
        
        np.testing.assert_array_equal(winner, [1, 3])
    
    
    def test_borda_count(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1)
        
        np.testing.assert_array_equal(points, [20, 15, 20,  9, 11])
        
    def test_borda_count_char(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1_char)
        
        np.testing.assert_array_equal(winner, ["A", "C"])
    
    def test_borda_count_char_winner_index(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1_char)
        
        np.testing.assert_array_equal(winner_index, [0, 2])
    
    
    def test_borda_count_example1a_points(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1a_char)
        
        np.testing.assert_array_equal(points, [29, 21, 27, 12, 16])
    
    def test_borda_count_example1a_winner(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1a_char)
        
        np.testing.assert_array_equal(winner, "A")
    
    def test_borda_count_example1a_winner_index(self):
        (candidates, points, (winner, winner_index)) = voting.borda_count(example1a_char)
        
        np.testing.assert_array_equal(winner_index, 0)
    
    #################
    # Coombs method #
    #################
    
    def test_coombs_ties(self):
        winner, votes = voting.coombs_method([[1, 2], [1, 2]])
        np.testing.assert_equal(winner, 1) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Use real example
        
    def test_coombs_ex1(self):
        winner, removed_actions = voting.coombs_method(example1)
        np.testing.assert_equal(winner, 1)
    
    def test_coombs_ex1_1(self):
        winner, removed_actions = voting.coombs_method(example1 - 1)
        np.testing.assert_equal(winner, 0)
    
    def test_coombs_character(self):
        winner, removed_actions = voting.coombs_method(example1_char)
        np.testing.assert_equal(winner, "A")
    
    ####################
    # Alternative vote #
    ####################


