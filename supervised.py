# Install required dependencies (Run this in Google Colab)
!pip install gradio scikit-learn pandas numpy scipy

import numpy as np
import pandas as pd
import gradio as gr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler

# Generating synthetic traffic data
data = {
    'Time': np.random.randint(0, 24, 100),  # Hour of the day
    'Speed': np.random.randint(10, 80, 100)  # Vehicle speed in km/h
}
df = pd.DataFrame(data)

# Scaling the data for better clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Applying Hierarchical Clustering
linkage_matrix = linkage(data_scaled, method='ward')
df['Cluster'] = fcluster(linkage_matrix, t=3, criterion='maxclust')

# Mapping clusters to traffic levels
traffic_labels = {1: 'Low Traffic', 2: 'Medium Traffic', 3: 'High Traffic'}
def predict_traffic(time, speed):
    input_data = scaler.transform([[time, speed]])
    input_linkage = linkage(np.vstack([data_scaled, input_data]), method='ward')
    cluster = fcluster(input_linkage, t=3, criterion='maxclust')[-1]
    return traffic_labels.get(cluster, 'Unknown')

# Gradio UI
demo = gr.Interface(
    fn=predict_traffic,
    inputs=[gr.Slider(0, 23, label='Hour of the Day'), gr.Slider(10, 80, label='Speed (km/h)')],
    outputs="text",
    title="Traffic Prediction using Hierarchical Clustering",
    description="Enter the hour and speed to predict traffic conditions."
)

demo.launch()
