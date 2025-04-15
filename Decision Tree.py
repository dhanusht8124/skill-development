# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Step 2: Load the dataset (using California Housing Dataset)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

# Convert to pandas DataFrame
df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Add target variable (house price)
df['Target'] = data.target

# Step 3: Data Exploration

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Basic statistics for numerical features
print("\nBasic Statistics for Features:")
print(df.describe())

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Step 4: Visualize the distribution of target variable (house prices)
plt.figure(figsize=(8, 6))
sns.histplot(df['Target'], bins=50, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Frequency")
plt.show()

# Step 5: Visualize the relationship between features and target (price)

# Scatter plot for 'AveRooms' vs 'Target' (house price)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['AveRooms'], y=df['Target'])
plt.title("Average Rooms vs House Price")
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.show()

# Step 6: Split the data into features (X) and target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Step 7: Train-test split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100)
}

# Dictionary to store model performance metrics
model_performance = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    model_performance[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

# Print model performance comparison
print("\nModel Performance Comparison:")
for model_name, metrics in model_performance.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2f}")

# Step 10: Visualize model performance (bar plot)
model_names = list(model_performance.keys())
r2_scores = [metrics['R2'] for metrics in model_performance.values()]

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=r2_scores)
plt.title("Model Comparison - R2 Scores")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.show()

# Step 11: Prediction and Error Distribution
# Let's use the best model (Random Forest Regressor) for residual analysis
best_model = models['Random Forest Regressor']
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

# Plot the residuals
residuals = y_test - y_pred_best

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=50)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Step 12: True vs Predicted House Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', lw=2)
plt.xlabel('True House Prices')
plt.ylabel('Predicted House Prices')
plt.title('True vs Predicted House Prices (Random Forest)')
plt.show()
