import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt




class MultiVariate_TS_EDA():
    def __init__(self, config, mv_ts):
        self.seq_identifier_col = config['seq_identifier_col'] 
        self.time_step_col = config['time_step_col'] 
        self.mv_ts = mv_ts
    
    def check_nans(self):
        nans = self.mv_ts.isna().sum().reset_index()
        return nans[nans[0]!=0]
    
    def impute_nans(self, method='median'):
        if method == 'median':
            self.mv_ts.fillna(self.mv_ts.median(), inplace=True)
        elif method == 'mean':
            self.mv_ts.fillna(self.mv_ts.mean(), inplace=True)
        else:
            raise ValueError("Invalid method. Choose either 'mean' or 'median'")
                
    
    def check_duplicates(self):
        duplicates = self.mv_ts.duplicated().sum()
        self.mv_ts.drop_duplicates(inplace=True)
        return duplicates
    
    def seq_lens_distrib(self):
        seq_lens = self.mv_ts.groupby(self.seq_identifier_col).size()
        plt.figure(figsize=(10, 5))
        sns.histplot(seq_lens, bins=10)
        ax = sns.histplot(seq_lens, kde=True, bins=15, color="blue")

        # Add a vertical line at the median
        ax.axvline(np.median(seq_lens), color="red", linestyle="--", linewidth=2, label=f"Median seq len: {np.median(seq_lens):.2f}")
        ax.set_title("Distribution of Sequence Lengths")
        ax.legend()
    
    def ts_decomopose(self, column):
        grouped_median = self.mv_ts.groupby([self.time_step_col, self.seq_identifier_col])[column].median().reset_index()
        result = seasonal_decompose(grouped_median[column], period=12*30*12, model='additive') # 12*30*24 = monthly, daily, bi-hourly
        return result
    
    

    # Function to plot based on dropdown selection
    def plot_ts_data(self, column):
        sns.set_theme(style="whitegrid")  # Optional: Set a consistent theme
        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(
            x=self.time_step_col, 
            y=column, 
            data= self.mv_ts, 
            marker='o', 
            label=column)
        
        ax.set_title(f"Trend for {column}", fontsize=14)
        ax.set_xlabel(self.time_step_col, fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()


        result = self.ts_decomopose(column)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
        result.trend.plot(ax=axes[0])
        result.seasonal.plot(ax=axes[1])
        result.resid.plot(ax=axes[2])

        axes[0].set_title(f"Trend for {column}", fontsize=14)
        axes[1].set_title(f"Seasonality for {column}", fontsize=14)
        axes[2].set_title(f"Residual for {column}", fontsize=14)
        # axes[1].set_title(f"Time-sereis Seasonality and trend decomoposition for {column}", fontsize=14)
        

    
