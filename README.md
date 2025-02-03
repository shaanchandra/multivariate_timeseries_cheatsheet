# Multivariate Timeseries Cheatsheet

This repository contains data preprocessing and standard EDA tools for multi-variate time-series data (panel data) to get started with quick EDA on any new dataset that one may receive.
It also optionally provides the framework to analyze any associated meta-data files related to the timesereis data (such as raw materials used in each batch, patient demographics, etc)


## Steps to use the tool
The usage of the tool is very simple. 

1. Place your data in the `./data` folder.
2. Open `mv_ts_EDA.ipynb` and edit the `config` dict at the top
3. Press RunAll for sequential execution of all the analysis.

## Features 

The following series of steps are performed in the end-to-end run of the NB:

1. **Read Data** : Read data (seq + static) and display basic statistics  

2. **Missing Value Treatment:**  Check for NaNs and treat them. Two treatment options are available:  
   2.1 **Drop** (`drop`): Drops all the rows with NaNs. If the percentage of NaNs is low, this can be used. Otherwise, significant data size reduction may occur.  
   2.2 **Impute** (`impute`) : Imputes the NaNs using either ***Median***  or ***Mean***  

3. **Duplicates Removal:**  If any duplicate rows exist, they are dropped.  

4. **Outlier Detection** : Identifies outliers in the target variable. Removes all data related to those identifiers (i.e., from both static and sequential data).  Two detection methods are provided:  
   4.1 **IQR-based Removal** (`iqr`) : Removes outliers based on the inter-quartile range.  *To Do: Add an option to change IQR limits (currently fixed at 99%).*  
   4.2 **Z-score-based Removal** (`z-score`)  : TBD

6.  

