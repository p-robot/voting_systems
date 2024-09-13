# Vote-processing rules for combining control recommendations from multiple models
![Tests](https://github.com/p-robot/voting_systems/actions/workflows/tests.yml/badge.svg?branch=master)
![Python version](https://img.shields.io/badge/python-3.8-blue.svg)


Vote-processing rules to accompany the publication entitled [*Vote-processing rules for combining rankings of control interventions from multiple models*](https://royalsocietypublishing.org/doi/10.1098/rsta.2021.0314).  This repo includes code for implementing the vote-processing rules, functions for translating model output into "votes", and several tests for all functions in this repository.  


The repository for implementing the analysis that is presented in the manuscript above is found at [voting_systems_epi_analysis](https://github.com/p-robot/voting_systems_epi_analysis).  It depends on having this module installed as it uses the functions defined herein.  

## Usage

This module was built using Python version 3.8.  The file [`requirements.txt`](voting_systems/tests/requirements.txt) provides a list of Python modules upon which this module depends and a virtual environment can be set up, activated, and the requirements for this module installed in the following manner: 

```bash
python -m venv venv
source venv/bin/activate
pip install -r voting_systems/tests/requirements.txt
```
Deactivate the environment using `deactivate`.

## Example usage

Consider a group of individuals ranking their favourite planets amongst the three closest to the sun (i.e. Mercury, Venus, Earth).

```python
import voting_systems as voting

# Define an example set of ballots; rows are 'voters'
# order is the preference; for instance, the first
# voter thought Mercury is the best, followed by Earth
# then Venus
example_planets = np.array([
    ["Mercury", "Earth",   "Venus"],
    ["Mercury", "Earth",   "Venus"],
    ["Mercury", "Venus",   "Earth"],
    ["Earth",   "Venus",   "Mercury"],
    ["Earth",   "Venus",   "Mercury"],
    ["Earth",   "Venus",   "Mercury"],
    ["Venus",   "Mercury", "Earth"]
])

# Process the ballots using Coombs method
(winner, winner_index), (candidates, removed) = voting.coombs_method(example_planets)
print(winner)
> "Earth"

# Process the ballots using Alternative vote method
(winner, winner_index), (candidates, removed) = voting.coombs_method(example_planets)
print(winner)
> "Mercury"
```

## Tests 

All vote processing-rules are tested using [`pytest`](https://docs.pytest.org/en/stable/).  Tests can be run in the following manner if called from this folder (after creating and activating a virtual environment): 

```bash
pytest
```

All tests are in [test_voting_rules.py](voting_systems/tests/test_voting_rules.py).  

## Publication

The publication associated with this code is the following:

- Probert et al., (2022) Vote-processing rules for combining control recommendations from multiple models.  Phil. Trans. R. Soc. A. 380: 20210314. https://doi.org/10.1098/rsta.2021.0314.
