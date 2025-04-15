!pip install gradio scikit-learn pandas matplotlib --quiet  # Install dependencies

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to perform K-Means clustering and visualize results
def kmeans_clustering(file, n_clusters):
    # Read the uploaded CSV file
    df = pd.read_csv(file.name)

    # Select only numerical columns for clustering
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return "Dataset must have at least two numerical columns!"

    # Standardize data (optional but recommended)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Plot the first two features with clusters
    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.7)
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])
    plt.title(f'K-Means Clustering with {n_clusters} Clusters')
    plt.colorbar(scatter, label="Cluster")

    # Save the plot
    plot_path = "cluster_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# Create a Gradio interface
iface = gr.Interface(
    fn=kmeans_clustering,
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Slider(2, 10, step=1, value=3, label="Number of Clusters (K)")
    ],
    outputs=gr.Image(type="filepath", label="Cluster Visualization"),  # Fixed 'type' issue
    title="K-Means Clustering with Gradio",
    description="Upload a CSV file with numerical data and choose the number of clusters to perform K-Means clustering."
)

# Launch the app
iface.launch()
