import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import random
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
        self.config = config
        self.seq_identifier_col = config['seq_identifier_col'] 
        self.time_step_col = config['time_step_col'] 
        self.mv_ts = mv_ts
        self.meta_data = meta_data
        self.ts_cols = config['ts_cols']
    
    
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


    def univariate_analysis_kde(self):        
        num_plots = len(self.ts_cols)
        num_rows = (num_plots + 4) // 5
        fig, axes = plt.subplots(num_rows, 5, figsize=(12, 2*num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(self.ts_cols):
            ax = axes[i]
            sns.kdeplot(data=self.mv_ts[feature], ax=ax)
            ax.set_title(f"KDE Plot - {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Density")
        for j in range(num_plots, num_rows * 3):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()


        
    def univariate_analysis_box(self):
        num_plots = len(self.ts_cols)
        num_rows = (num_plots + 4) // 5 
        fig, axes = plt.subplots(num_rows, 5, figsize=(12, 2*num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(self.ts_cols):
            ax = axes[i]
            sns.boxplot(data =self.mv_ts, x=self.mv_ts[feature], ax=ax)
            ax.set_title(f"Box Plot - {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Value")

        for j in range(num_plots, num_rows * 3):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()
    

    def random_sumsample_time_series_plot(self, n_samples=10):
        # df = self.data
        
        
        random.seed(41)
        l = random.sample(list(self.mv_ts[self.config['seq_identifier_col']].unique()), n_samples)
        print(">> Plotting for the following batches:  ", l)
        
        df = self.mv_ts[self.mv_ts[self.config['seq_identifier_col']].isin(l)]
        df = df.sort_values(by = self.config['time_step_col'])
        df.set_index(self.config['time_step_col'], inplace=True)

        num_plots = len(self.ts_cols)
        num_rows = (num_plots + 4) // 3
        fig, axes = plt.subplots(num_rows, 3, figsize=(25, 3*num_rows))
        axes = axes.flatten()

        for i, feature in enumerate(self.ts_cols):
            ax = axes[i]            
            df.groupby(self.config['seq_identifier_col'])[feature].plot(ax = ax)
            ax.set_title(f"Plot of- {feature}")
            # ax.set_xlabel(feature)
            ax.set_ylabel("Value")
            ax.tick_params(labelrotation=45, labelsize = 7)

        for j in range(num_plots, num_rows * 3):
            fig.delaxes(axes[j])            
        plt.tight_layout()
        plt.show()

    
    

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
        plt.figure(figsize=(12, 4))
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

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 5))
        result.trend.plot(ax=axes[0])
        result.seasonal.plot(ax=axes[1])
        result.resid.plot(ax=axes[2])

        axes[0].set_title(f"Trend for {column}", fontsize=14)
        axes[1].set_title(f"Seasonality for {column}", fontsize=14)
        axes[2].set_title(f"Residual for {column}", fontsize=14)



    # Function to plot ACF, PACF
    def plot_acf_pacf(self, column):
        """
        Plots the autocorrelation and partial autocorrelation functions for a specified column.
        Parameters
        ----------
        column : str
            The column name (from dropdown selection) denoting the time-series to plot ACF and PACF for.
        """
        fig, axes = plt.subplots(1, 2, sharex=False, figsize=(25, 5))
        plot_acf(self.mv_ts[column], ax=axes[0], lags=100, alpha=0.05, auto_ylims=True, title=f"ACF for {column}")
        plot_pacf(self.mv_ts[column], ax=axes[1], lags=100, alpha=0.05, auto_ylims=True, title=f"PACF for {column}")
        plt.show()

    
    def agg_vif(self):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.ts_cols
        vif_data["VIF"] = [variance_inflation_factor(self.mv_ts[self.ts_cols].values, i) for i in range(len(self.ts_cols))]
        vif_data = vif_data.sort_values(by = 'VIF', ascending = False)
        print(vif_data)
    

    # Function to calculate VIF for all variables
    def plot_pairwise_vif(self):
        variables = self.ts_cols
        vif_matrix = pd.DataFrame(index=variables, columns=variables)
        
        for i, var_i in enumerate(variables):
            for j, var_j in enumerate(variables):
                if i == j:
                    vif_matrix.loc[var_i, var_j] = np.nan  # Diagonal is not applicable
                else:
                    # Calculate VIF for var_i regressed on var_j
                    X = self.mv_ts[[var_j]]
                    y = self.mv_ts[var_i]
                    vif = variance_inflation_factor(np.column_stack((X, y)), 1)
                    vif_matrix.loc[var_i, var_j] = vif
        
        vif_matrix = vif_matrix.astype(float)
        # Plot the heatmap
        plt.figure(figsize=(18, 12))
        sns.heatmap(vif_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
        plt.title('Pairwise Variance Inflation Factor (VIF) Heatmap')
        plt.show()




    
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

        

    
