import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




class Multivariate_TS_Clustering:
    """
    A class used to perform clustering on multivariate time series data.

    Attributes
    ----------
    mv_ts : pd.DataFrame
        The multivariate time series data.
    meta_data : pd.DataFrame
        The metadata associated with the time series data.
    """
    def __init__(self, mv_ts, meta_data):
        self.mv_ts = mv_ts
        self.meta_data = meta_data




    def KMeans_cluster_meta_data(self, n_clusters=2):
        """
        Clusters the metadata associated with the time series data.

        Attributes
        ----------
        n_clusters : int, optional
            The number of clusters to create. Default is 3.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metadata with cluster labels.
        KMeans
            The KMeans clustering model.
        """
        # Standardize the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.meta_data.select_dtypes(include=[np.number]))

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.scaled_data)

        # Add cluster labels to the original dataframe
        self.meta_data['Cluster'] = clusters
        return self.meta_data, kmeans
    




    def perform_pca_and_plot(self, n_components=2):  
        """
        Perform PCA on the cluster assignments and plot the results.

        Attributes
        ----------
        n_components (int): 
            Number of principal components to compute. Default is 2.
        
        Returns:
        ----------
        None
        """
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.scaled_data)

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['Cluster'] = self.meta_data['Cluster']

        # Plot the PCA results using seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.7, edgecolor='k')
        plt.title('PCA of Clustered Meta Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.show()



        
    

    def visualize_cluster_differences(self):
        """
        Visualizes the differences in a specified column across different clusters.

        Parameters
        ----------
        column : str
            The column name to visualize.
        """
        # Visualize batch_stats as box plots

        plt.figure(figsize=(20, 10))
        meta_data_clustered_melted = self.meta_data.melt(id_vars='Cluster', var_name='Metric', value_name='Value')
        metrics = meta_data_clustered_melted['Metric'].unique()
        num_metrics = len(metrics)
        num_cols = 4
        num_rows = (num_metrics // num_cols) + (num_metrics % num_cols > 0)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            sns.boxplot(x='Cluster', y='Value', data=meta_data_clustered_melted[meta_data_clustered_melted['Metric'] == metric], ax=axes[i], palette='Set2', hue='Cluster')
            axes[i].set_title(metric)
            axes[i].set_xlabel('Cluster')
            axes[i].set_ylabel('Value')

        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
