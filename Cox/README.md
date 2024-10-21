# Practical_debiasing/Cox

This repository contains the scripts used in the manuscript "Practical debiasing with the Covariant Prior in the proportional regime when $p<n$".


| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|```required_functions```| Collections of .py scripts that is used to generate the data, fit the Cox model with Covariant Prior and solve the Replica Symmetric (RS) equations of the theory, as explained in the reference manuscript. | 
|```run_sim_histograms```| Script used to compute the data for the histograms in figure $7$ for the Cox model in the reference manuscript. It returns a csv file names 'sim_thetas_...' containing $m = 500$ realization of the estimator for $\theta_0$.|
|```run_sim_order_parameters```| Script used to compute the data for the plots in figure $6$ in the reference manuscript. It returns: 1)  a csv file named 'sim_zeta...'containing the average and the standard deviation (over $m =50$ realizations) of the RS order parameters along the regularization path and 2) a csv file named 'rs_zeta...' containing the solution of the RS order parameters along the regularization path.   |
|```order_paramters_plots ```| Script that generates the plots in figure $6$ in the reference manuscript. |
| ```qq_plot ```| Script used to generate the plots in figures $8$ and $9$ in the reference manuscript. |
| ```histograms_plots ```|Script used to generate the plots in figure $7$ in the reference manuscript.  |
|```tutorial ```| Tutorial explaining the use of the classes in "required_functions" used throughout the previous scripts : it showcases how to generate survival data, fit the Cox regression model with covariant regularization, estimate the RS order parameters from the data and solve the RS equations.|    