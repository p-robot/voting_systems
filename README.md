# `voting_systems`

Vote-processing rules to accompany the publication entitled *Vote-processing rules for combining rankings of control interventions from multiple models*.  This repo includes code for implementing the vote-processing rules, functions for translating model output into "votes", and several tests for all functions in this repository.  


The repository for implementing the analysis that is presented in the manuscript above is found at [voting_systems_epi_analysis](https://github.com/p-robot/voting_systems_epi_analysis).  It depends on having this module installed as it uses the functions defined herein.  


### Virtual environment

[`voting_systems/tests/requirements.txt`](voting_systems/tests/requirements.txt) provides a list of Python modules upon which this module depends.  


We recommend running and testing this code in a virtual environment.  A Python virtual environment can be set up, activated, and the requirements for this module installed in the following manner: 

```bash
python -m venv venv
source venv/bin/activate
pip install -r voting_systems/tests/requirements.txt
```

The Python virtual environment can be deactivated by simply typing `deactivate`.  

### Tests 

All vote processing-rules are tested using [`pytest`](https://docs.pytest.org/en/stable/).  Tests can be run in the following manner if called from this folder (after creating and activating a virtual environment): 

```bash
pytest
```

All tests are in [test_voting_rules.py](voting_systems/tests/test_voting_rules.py).  

