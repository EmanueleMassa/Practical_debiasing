# Practical_debiasing

This repository contains the scripts used in the manuscript "".

In order to run the python scripts in the main directory you need to have 'routines' as a subdirectory. 

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```/routines/``` | Collections of .py scripts that is used to generate the data, fit the Cox model with Elastic Net regularization, solve the RS equations      |                               |
| ```tutorial_cox_model.ipynb``` | Notebook with worked example to generate a data-set and fit the elastic net regularized COX model either via COX-AMP or via Coordinate-wise Descent.                   |
| ```tutorial_cox_rs.ipynb``` | Notebook with worked example to solve the RS equations and compare with the estimates of the order parameters obtained from the data.         |
| ```run_experiment.py ```         | Script that : 1) generates 50 data-sets, with the specifics indicated in the reference manuscript, 2) fits the cox models along a regularization path and 3) estimates the order parameters solely from the data (along the regularization path). It is one of the routines used to obtain the plots in the folder figures.
| ```run_simple_experiment.py ```   | Script that : 1) generates 50 data-sets, with the specifics indicated in the reference manuscript, 2) fits the cox models along a regularization path and computes the overlaps $w$ and $v$. Notice that in this case the data-generating process is known and this information is used to compute $w,v$. It is one of the routines used to obtain the plots in the folder figures.    