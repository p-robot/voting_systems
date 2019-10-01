#!/usr/bin/env python3
"""
Script to test vote-processing rules against some examples

W. Probert, 2019
"""
import numpy as np, sys
sys.path.append('/Users/willprobert/Projects/voting_systems')

import voting_systems as voting


# Just to be clear, model 1 thought action A was the best, followed by D, then C, E, 
# and thought B was the worst action.  

example1 = np.array([
    [1, 4, 3, 5, 2],
    [1, 3, 5, 4, 2],
    [3, 2, 1, 5, 4],
    [3, 2, 1, 5, 4],
    [2, 1, 3, 5, 4]
])


example1_char = np.array([
    ["A", "D", "C", "E", "B"],
    ["A", "C", "E", "D", "B"],
    ["C", "B", "A", "E", "D"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"]
])


example2_char = np.array([
    ["D", "A", "C", "E", "B"],
    ["D", "C", "E", "A", "B"],
    ["C", "B", "A", "E", "D"],
    ["B", "A", "C", "E", "D"]
])


preferences = np.array([[5, 4, 3, 2, 1], [5, 3, 2, 1, 4], [1, 2, 3, 4, 5], [2, 1, 3, 5, 4]])
# Using this example: 
# 1) No overall majority
# 2) Action 1 should be removed first (it has the most 5's)
# 3) Any 1st preference votes for action 1 should be given to the action that that was voted second


preferences = np.array([[5, 4, 3, 2, 1], [5, 3, 2, 1, 4], [1, 2, 3, 4, 5], [1, 2, 5, 3, 4], \
    [4, 2, 3, 5, 1], [2, 1, 3, 5, 4] , [5, 3, 2, 1, 4]])



class TestClass(object):
    """
    Test class for checking 
    """
    @classmethod
    def setup_class(self):
        """
        Set-up method
        """
        pass
        
    @classmethod
    def teardown_class(self):
        """
        Tear-down method
        """
        pass
    
    def test_fpp_ties(self):
        winner, votes = voting.fpp(example1)
        np.testing.assert_equal(1, 1) # <<<<<<<<<<<<<<<<<<<<< fix this to allow ties
    
    def test_fpp_char(self):
        winner, votes = voting.fpp(example2_char)
        np.testing.assert_equal(winner, "D")
    
    def test_borda_count(self):
        winner, votes = voting.borda_count(example1)
        np.testing.assert_equal(winner, 1)
        
    def test_borda_count_char(self):
        winner, votes = voting.borda_count(example1_char)
        np.testing.assert_equal(winner, "A")
    
    def test_coombs_ties(self):
        winner, votes = voting.fpp([[1, 2], [1, 2]])
        np.testing.assert_equal(winner, 1)
        
    def test_coombs_ex1(self):
        winner, removed_actions = voting.coombs_method(example1)
        np.testing.assert_equal(winner, 1)
    
    def test_coombs_ex1_1(self):
        winner, removed_actions = voting.coombs_method(example1 - 1)
        np.testing.assert_equal(winner, 0)
    
    def test_coombs_character(self):
        winner, removed_actions = voting.coombs_method(example1_char)
        np.testing.assert_equal(winner, "A")
