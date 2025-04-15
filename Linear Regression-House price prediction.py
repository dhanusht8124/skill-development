import gradio as gr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing  # Importing the California housing dataset

# Load the California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

# Define the prediction function
def predict_house_price(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat):
    input_features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
    predicted_price = model.predict(input_features)
    return predicted_price[0]

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_house_price,
    inputs=[
        gr.Slider(minimum=0, maximum=100, step=0.1, label="Crime Rate (CRIM)"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Residential Area Proportion (ZN)"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Non-retail Business Acreage (INDUS)"),
        gr.Checkbox(label="Charles River Dummy Variable (CHAS)"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Nitrogen Oxides Concentration (NOX)"),
        gr.Slider(minimum=0, maximum=10, step=0.1, label="Average Number of Rooms (RM)"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Age of House (AGE)"),
        gr.Slider(minimum=0, maximum=10, step=0.1, label="Distance to Employment Centers (DIS)"),
        gr.Slider(minimum=0, maximum=24, step=1, label="Radial Highways Accessibility (RAD)"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Property Tax Rate (TAX)"),
        gr.Slider(minimum=0, maximum=30, step=1, label="Pupil-Teacher Ratio (PTRATIO)"),
        gr.Slider(minimum=0, maximum=1000, step=1, label="Proportion of Black Residents (B)"),
        gr.Slider(minimum=0, maximum=40, step=1, label="Lower Status Population Proportion (LSTAT)")
    ],
    outputs=gr.Textbox(label="Predicted House Price (in $1000s)"),
    live=True  # Live prediction while adjusting inputs
)

# Launch the Gradio interface
interface.launch()
