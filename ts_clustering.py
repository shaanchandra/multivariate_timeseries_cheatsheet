import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Flatten(),
            # nn.Linear(64 * (input_channels // 8), latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    

class CNNDecoder(nn.Module):
    def __init__(self, output_channels):
        super(CNNDecoder, self).__init__()
        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim, 64 * (output_channels // 8)),
            # nn.ReLU(),
            # nn.Unflatten(1, (64, output_channels // 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=2, padding=1),
            # nn.Sigmoid()
        )


    def forward(self, x):
        return self.decoder(x)
    


class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels):
        super(CNNAutoencoder, self).__init__()
        self.cnnencoder = CNNEncoder(input_channels)
        self.cnndecoder = CNNDecoder(input_channels)

    def forward(self, x):
        latent = self.cnnencoder(x)
        reconstructed = self.cnndecoder(latent)
        return reconstructed



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
    def __init__(self, config, mv_ts, meta_data):
        self.seq_identifier_col = config['seq_identifier_col'] 
        self.time_step_col = config['time_step_col'] 
        self.ts_cols = config['ts_cols']
        self.config = config

        self.mv_ts = mv_ts
        print(self.config)
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



    
    def plot_multivariate_ts_as_image(self, selected_batch_id):
        filtered_data = self.mv_ts[self.mv_ts[self.seq_identifier_col] == selected_batch_id]

        # Min-max scale the time-series values
        scaler = MinMaxScaler()
        # scaled_data = filtered_data.copy()
        filtered_data[self.ts_cols] = scaler.fit_transform(filtered_data[self.ts_cols])

        # Plot the heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(filtered_data[self.ts_cols].T)
        plt.title('Multi-variate Time-series Heatmap')
        plt.xlabel('Time Steps')
        plt.ylabel('Time-series Variables')
        plt.show()
    

    def convert_dataframe_to_3D_tensor(self):
        """
        Convert a DataFrame containing multi-variate sequences into a 3D PyTorch tensor.

        Args:
            df (pd.DataFrame): Input DataFrame containing sequences.
            sequence_col (str): Column name identifying sequence groups.
            feature_cols (list): List of column names to be treated as features.

        Returns:
            torch.Tensor: A 3D tensor of shape (num_sequences, num_columns, max_time_steps).
        """
        # Group the DataFrame by the sequence column
        grouped = self.mv_ts.groupby(self.seq_identifier_col)
        
        # Extract sequences and their lengths
        sequences = [group[self.ts_cols].values for _, group in grouped]
        lengths = [seq.shape[0] for seq in sequences]
        max_length = max(lengths)
        
        # Pad sequences with zeros
        padded_sequences = [np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant') for seq in sequences]
        
        # Convert to tensor and rearrange to (num_sequences, num_columns, num_time_steps)
        print(">> Max len seq for padding the rest = ", max_length)
        self.mv_ts_tensor = torch.tensor(padded_sequences).permute(0, 2, 1).float()
        print(">> The 3D tensor constructed (num_seqs, num_cols, max_time_steps) = ", self.mv_ts_tensor.size())




    def prepare_data(self): 
        # Prepare data
        print("-"*50 + "\nPreparing data for training the CNN Autoencoder\n" + "-"*50)
        self.convert_dataframe_to_3D_tensor()
        dataset = TensorDataset(self.mv_ts_tensor, self.mv_ts_tensor)
        self.dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
        print(">> Data prepared for training the CNN Autoencoder\n")
    

    def prepare_model_and_optmzr(self):
        # Initialize model, loss function, and optimizer
        print("-"*50 + "\nInitializing models and optimizers for training the CNN Autoencoder\n" + "-"*50)
        self.model = CNNAutoencoder(self.config['input_channels'])
        print(">> Model architecture: \n", self.model)
        print(">> Total trainable model parameters = ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        print(">> Training on MSE loss using Adam optimizer\n")
    
    def _save_model(self):
        # print("\n\n>> Saving model checkpoint at :   ", self.config['model_save_path'])
        # Save the model
        torch.save(self.model.state_dict(), self.config['model_save_path'])
    

    def train_CNN_AE(self):
        # Training loop
        losses = []
        lowest_loss = 5e40
        for epoch in range(self.config['num_epochs']):
            loss_list = []
            for data in self.dataloader:
                inputs, _ = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs.unsqueeze(1))
                # print(inputs.shape, outputs.shape)
                loss = self.criterion(outputs.squeeze(1), inputs)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
            
            epoch_loss = sum(loss_list)/len(loss_list)
            if epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_epoch = epoch
                self._save_model()

            
            if epoch % 20 == 0:
                if epoch_loss<500:
                    losses.append(epoch_loss)
                print(f'Epoch {(epoch+1)}, Loss: {epoch_loss:.4f}')
            
            if epoch%100==0:
                print(f">> Current best model at epoch {best_epoch} and loss {lowest_loss : .4f}")

        print(f'\n>> Final Epoch, Loss: {epoch_loss:.4f}')
        # plot the loss
        plt.plot(losses)
        plt.title("CNN-AE training loss")
        plt.legend()



