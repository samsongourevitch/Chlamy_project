import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
from scipy.stats import ttest_1samp
from scipy.stats import multivariate_normal
from scipy.special import comb
from scipy.special import erf
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from umap import UMAP

def get_clusters_whole_time_series(data, method, n_clusters=3, plot=False):
    data_cluster = data.copy()
    if method == 'pca':
        for light in data['light_regime'].unique():
            data_light = data[data['light_regime'] == light]
            data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
            # Create a PCA model with 2 components: pca
            pca = PCA(n_components=2)

            kmeans = KMeans(n_clusters=n_clusters)

            # Make a pipeline chaining normalizer, pca and kmeans: pipeline
            pipeline = make_pipeline(StandardScaler(), pca, kmeans)
            pipeline.fit(data_light_y2)

            # Calculate the cluster labels: labels
            labels_pca = pipeline.predict(data_light_y2)

            # add the labels to the data
            data_cluster.loc[data_cluster['light_regime'] == light, 'cluster_' + method] = labels_pca
            
            if plot == True:
                # Plot the clusters
                plt.scatter(data_light_y2[:, 0], data_light_y2[:, 1], c=labels_pca, cmap='viridis')
                plt.xlabel('PCA 1')
                plt.ylabel('PCA 2')
                plt.title('PCA clustering')
                plt.show()

    elif method == 't-sne':
        for light in data['light_regime'].unique():
            data_light = data[data['light_regime'] == light]
            data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
            
            tsne = TSNE(n_components=3)

            data_tsne = tsne.fit_transform(data_light_y2)

            kmeans = KMeans(n_clusters=n_clusters)

            pipeline = make_pipeline(kmeans)

            # Fit the pipeline to samples
            pipeline.fit(data_tsne)

            labels_tsne = pipeline.predict(data_tsne)

            data_cluster.loc[data_cluster['light_regime'] == light, 'cluster_' + method] = labels_tsne

            if plot == True:
                plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_tsne, cmap='viridis')
                plt.xlabel('TSNE 1')
                plt.ylabel('TSNE 2')
                plt.title('TSNE clustering')
                plt.show()

    elif method == 'umap':
        for light in data['light_regime'].unique():
            data_light = data[data['light_regime'] == light]
            data_light_y2 = data_light.filter(like='y2_').dropna(axis=1).values.astype(float)
            
            umap = UMAP(n_components=2)

            data_umap = umap.fit_transform(data_light_y2)

            kmeans = KMeans(n_clusters=n_clusters)

            pipeline = make_pipeline(kmeans)

            # Fit the pipeline to samples
            pipeline.fit(data_umap)

            labels_umap = pipeline.predict(data_umap)

            data_cluster.loc[data_cluster['light_regime'] == light, 'cluster_' + method] = labels_umap

            if plot == True:
                # Plot the UMAP embedding
                plt.figure(figsize=(8, 6))
                plt.scatter(data_umap[:, 0], data_umap[:, 1], c=labels_umap, cmap='viridis')
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')
                plt.title('UMAP Embedding Clustering')
                plt.show()
    
    return data_cluster

def plot_clusters(data, method, n=50):
    n_clusters = len(data['cluster_' + method].unique())
    fig, axs = plt.subplots(nrows=(n_clusters - 1)//3 + 1, ncols=3, figsize=(15, 5*(n_clusters - 1)//3 + 1))
    for i, cluster in enumerate(data['cluster_' + method].unique()):
        data_cluster = data[data['cluster_' + method] == cluster]
        data_cluster_y2 = data_cluster.filter(like='y2_').dropna(axis=1).values.astype(float)
        if n_clusters <= 3:
            axs[i].plot(data_cluster_y2[:n].T, alpha=0.5)
            axs[i].set_title('Cluster ' + str(i))
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Y')
        else :
            axs[i//3, i%3].plot(data_cluster_y2[:n].T)
            axs[i//3, i%3].set_title('Cluster ' + str(cluster))
            axs[i//3, i%3].set_xlabel('Time')
    plt.tight_layout()
    plt.show()
        
    