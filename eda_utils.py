import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
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
    def __init__(self, config, mv_ts, meta_data):
        self.seq_identifier_col = config['seq_identifier_col'] 
        self.time_step_col = config['time_step_col'] 
        self.mv_ts = mv_ts
        self.meta_data = meta_data
    
    
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



    def treat_target_outliers(self, method='iqr'):
        """
        Treats outliers in the target variable of time series data using the specified method.

        Parameters
        ----------
        method : str, optional
            The method to use for outlier treatment ('iqr' or 'z-score'). Default is 'iqr'.
        """
    
        batch_ids_before = self.mv_ts[self.seq_identifier_col].unique()
        size_before = self.mv_ts.shape[0]

        if method == 'iqr':
            Q1 = self.mv_ts['5K_VCC'].quantile(0.005)
            Q3 = self.mv_ts['5K_VCC'].quantile(0.999)
            IQR = Q3 - Q1
            self.mv_ts = self.mv_ts[~((self.mv_ts['5K_VCC'] < Q1) | (self.mv_ts['5K_VCC'] > Q3))]
        elif method == 'z-score':
            z = np.abs(stats.zscore(self.mv_ts['5K_VCC']))
            self.mv_ts['5K_VCC'] = self.mv_ts['5K_VCC'][(z < 3).all(axis=1)]
        else:
            raise ValueError("Invalid method. Choose either 'iqr' or 'z-score'")
        
        batch_ids_after = self.mv_ts[self.seq_identifier_col].unique()

        if self.mv_ts.shape[0] == size_before:
            print("\n\n>> No outliers found in the target variable...")
        else:            
            print(f"\n\n>> IQR range for target variable = {Q1} to {Q3}")
            print(f">> {size_before - self.mv_ts.shape[0]} outlier rows found in target variable, removed using {method} method...")
            print(f">> Total batches removed = {len(batch_ids_before) - len(batch_ids_after)}")
    

    
    def seq_lens_distrib(self):
        """
        Plots the distribution of sequence lengths in the time series data along with marking the median length.
        """
        seq_lens = self.mv_ts.groupby(self.seq_identifier_col).size()
        plt.figure(figsize=(8, 4))
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



    
    def batch_start_time_corr_with_yield(self, cluster_plots=False):
        """
        Analyzes the correlation between batch start time and yield (5K_VCC) by plotting and calculating rolling means.
        
        Returns:
            pd.DataFrame: A DataFrame containing the merged data with batch start times and 5K_VCC values.
        """

        # Get the rows with the highest TIME_STEP for each BATCH_ID
        max_time_step_df = self.mv_ts.loc[self.mv_ts.groupby(self.seq_identifier_col)[self.time_step_col].idxmax(), [self.seq_identifier_col, 'START_TIME', '5K_VCC']]

        # Merge with meta_data
        self.merged_df = pd.merge(self.meta_data, max_time_step_df, on=self.seq_identifier_col)
        self.merged_df['START_TIME'] = pd.to_datetime(self.merged_df['START_TIME']).dt.strftime('%d-%m-%y')

        # Plot the correlation between batch start time and yield
        plt.figure(figsize=(35, 10))
        sns.lineplot(data=self.merged_df, x='START_TIME', y='5K_VCC', marker='o')
        # Calculate rolling mean
        rolling_mean1 = self.merged_df['5K_VCC'].rolling(window=10).mean()
        rolling_mean2 = self.merged_df['5K_VCC'].rolling(window=50).mean()
        # Plot the rolling mean
        sns.lineplot(data=self.merged_df, x='START_TIME', y=rolling_mean1, label='Rolling Mean - 10', linewidth=3)
        sns.lineplot(data=self.merged_df, x='START_TIME', y=rolling_mean2, label='Rolling Mean - 50', linewidth=3)
        plt.title('5K_VCC over Time')
        plt.xlabel('Batch Start Time')
        plt.ylabel('5K_VCC')
        plt.xticks(rotation=90)

        if cluster_plots:
            # Plot the correlation between batch start time and yield for each cluster
            for cluster in self.merged_df['Cluster'].unique():
                cluster_df = self.merged_df[self.merged_df['Cluster'] == cluster]
                sns.scatterplot(data=cluster_df, x='START_TIME', y='5K_VCC', marker='x', s=200, linewidth=3, label=f'Cluster {cluster}')
            plt.title('5K_VCC over Time by Cluster')
            plt.xlabel('Batch Start Time')
            plt.ylabel('5K_VCC')
            plt.xticks(rotation=90)
            plt.legend(title='Cluster')
        plt.show()

        

    
