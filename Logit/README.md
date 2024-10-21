# Practical_debiasing/Logit

This repository contains the scripts used in the manuscript "Practical debiasing with the Covariant Prior in the proportional regime when $p<n$".


| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|```required_functions```| Collections of .py scripts that is used to generate the data, fit the Logit model with Covariant Prior and solve the Replica Symmetric (RS) equations of the theory, as explained in the reference manuscript. | 
|```run_sim_histograms```| Script used to obtain the histograms in figure $4$ for the Logit model in the reference manuscript. |
|```run_sim```| Script used to compute the data for the plots in figures $2$ and $3$ in the reference manuscript.
It returns a csv file named 'sim_zeta...' containing the average and the standard deviation (over $m =50$ realizations) of the RS order parameters along the regularization path. |
|```run_rs```| Script used to compute the data for the plots in figure $2$ and $3$ in the reference manuscript. 
It returns a csv file named 'rs_zeta...' containing the solution of the RS equations along the regularization path. |
| ```qq_plot ```| Script used to generate the plots in figure $5$ in the reference manuscript. |
| ```plots ```|Script used to generate the plots in figures $2$ and $3$ in the reference manuscript.  |
|```tutorial ```| Tutorial explaining the use of the classes in "required_functions" used throughout the previous scripts : it showcases how to generate data from a logit model, fit the Logit regression model with covariant regularization, estimate the RS order parameters from the data and solve the RS equations.|    
