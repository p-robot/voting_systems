# `voting_systems`

Code to accompany the publication entitled *Vote-processing rules for combining rankings of control interventions from multiple models*.  This repo includes code for implementing vote-processing rules.  



### Analyses

There are three main analyses in this repository: 

* **Analysis 1:** Applies four vote-processing rules (First-past-the-post, Alternative Vote, Coombs method, Borda Count) to two case studies with multiple models (Probert et al., 2016; Li et al., 2017).  

* **Analysis 2:** Sensitivity analysis of biased models.  

* **Analysis 3:** Sensitivity analysis of including random models.  




### Virtual environment

`requirements.txt` provides a list of Python modules upon which this module depends.  


### Tests 

All vote processing-rules are tested using [`pytest`](https://docs.pytest.org/en/stable/).  Tests can be run in the following manner: 

```bash
pytest
```
