"""Voting systems for decision-making with multiple models"""
name = "voting_systems"
__version__ = '0.1'

from .core import fpp, borda_count, coombs_method, alternative_vote
from .core import values_to_votes
