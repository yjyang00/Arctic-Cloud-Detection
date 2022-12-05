Sta521 - Fall 2022 - Project 2: Arctic Cloud Detection
--------
Authors: Huiying Lin, Yanjiao Yang

### Overview

This project conducts classification methods to detect the presence of cloud from satellite pictures in the polar regions. The data file contains the original three images which are obtained by the MISR sensor abroad the NASA satellite Terra. We start by basic exploration data analysis, which includes a brief summary of pixel labels, visualizations of well-labeled pixels and correlation between features. Then we split the data in two ways and generalize the cross validation process by writing a `CVmaster` function. As for training the model, we fit seven classification methods, assess the performance for each, and select the random forest as the best model for an in-depth diagnostics. In diagnostics, we present convergence plot, variable importance plots, etc. and report the results for both splits. Underlying misclassification patterns and improved classifier are proposed at the end.

---
### Reproducibility
To reproduce our results, we provide the following files:  

*  `data`: the three images obtained by the MISR sensor
* `PROJ2-writeup.tex`: the raw Latex used to generate the report  
* `PROJ2-code.rmd`: the code written for all parts. One can get all the figures and plots by running the code chunk by chunk. In detail, part 1 is to load data, plot labeled maps and perform EDA, part 2 is to do data split, calculate baseline accuracy and find the best features, part 3 is to compare different models using multiple ways, and part 4 is to further diagnose the random forest model.  
* `CVmaster.R`: the generic cross validation function that outputs the K-fold corss validation loss on the training set.  
