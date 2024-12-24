import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt




class MultiVariate_TS_EDA():
    """
    A class used to perform exploratory data analysis (EDA) on multivariate time series data.

    Attributes
    ----------
    config : dict
        The main config dictionary containing the user settings.
    time_step_col : str
        The column name that represents the time steps in the time series data.
    mv_ts : pd.DataFrame
        The multivariate time series data.
    """
    def __init__(self, config, mv_ts):
        self.seq_identifier_col = config['seq_identifier_col'] 
        self.time_step_col = config['time_step_col'] 
        self.mv_ts = mv_ts
    
    
    def treat_nans(self, treatment='impute'):
        """
        Checks for NaN values in the time series data.

        Attributes
        ----------
        treatment : str, optional
            The treatment to apply to NaN values ('impute' or 'drop'). Default is 'impute'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the count of NaN values for each column that has NaNs.
        """
        nans = self.mv_ts.isna().sum().reset_index()
        nans_counts = nans[nans[0]!=0]

        if len(nans_counts)>0:
            print(">> NaNs in the data:  ", nans_counts)
            if treatment == 'impute':
                print(">> Imputing NaNs...")
                self.impute_nans()
            else:
                print(">> Dropping NaNs...")
                self.mv_ts.dropna(inplace=True)
        else:
            print(">> No NANs in the data...")
        return nans_counts
    


    
    def impute_nans(self, method='median'):
        """
        Imputes NaN values in the time series data using the specified method.

        Parameters
        ----------
        method : str, optional
            The method to use for imputation ('mean' or 'median'). Default is 'median'.
        """
        if method == 'median':
            self.mv_ts.fillna(self.mv_ts.median(), inplace=True)
        elif method == 'mean':
            self.mv_ts.fillna(self.mv_ts.mean(), inplace=True)
        else:
            raise ValueError("Invalid method. Choose either 'mean' or 'median'")
                
    

    def treat_duplicates(self):
        """
        Checks for duplicate rows in the time series data and removes them.
        Returns
        -------
        int
            The number of duplicate rows found and removed.
        """
        duplicates = self.mv_ts.duplicated().sum()
        self.mv_ts.drop_duplicates(inplace=True)
        if duplicates==0:
            print("\n\n>> No duplicates in the data...")
        else:
            print(">> Duplicates in the data:  ", duplicates)
            print(">> Duplicates removed from the data...")
            print(">> New shape of the data:  ", self.mv_ts.shape)
        return duplicates
    

    
    def seq_lens_distrib(self):
        """
        Plots the distribution of sequence lengths in the time series data along with marking the median length.
        """
        seq_lens = self.mv_ts.groupby(self.seq_identifier_col).size()
        plt.figure(figsize=(10, 5))
        sns.histplot(seq_lens, bins=10)
        ax = sns.histplot(seq_lens, kde=True, bins=15, color="blue")

        # Add a vertical line at the median
        ax.axvline(np.median(seq_lens), color="red", linestyle="--", linewidth=2, label=f"Median seq len: {np.median(seq_lens):.2f}")
        ax.set_title("Distribution of Sequence Lengths")
        ax.legend()



    
    def ts_decomopose(self, column):
        """
        Decomposes the time series data for a specified column into trend, seasonal, and residual components.
        Parameters
        ----------
        column : str
            The column name (from dropdown selection) denoting the time-series to decompose.
        Returns
        -------
        result
            The result of the seasonal decomposition (trend, seasonality and residuals).
        """
        grouped_median = self.mv_ts.groupby([self.time_step_col, self.seq_identifier_col])[column].median().reset_index()
        result = seasonal_decompose(grouped_median[column], period=12*30*12, model='additive') # 12*30*24 = monthly, daily, bi-hourly
        return result
    
    

    # Function to plot based on dropdown selection
    def plot_ts_data(self, column):
        """
        Plots the time series data and its decomposed components for a specified column.
        Parameters
        ----------
        column : str
            The column name (from dropdown selection) denoting the time-series to plot.
        """

        # Plot the time-series for selected time-series
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


        # Plot the time-series decomposition plots of the selected time-sereis
        result = self.ts_decomopose(column)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
        result.trend.plot(ax=axes[0])
        result.seasonal.plot(ax=axes[1])
        result.resid.plot(ax=axes[2])

        axes[0].set_title(f"Trend for {column}", fontsize=14)
        axes[1].set_title(f"Seasonality for {column}", fontsize=14)
        axes[2].set_title(f"Residual for {column}", fontsize=14)
        

    
