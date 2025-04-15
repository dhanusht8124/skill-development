import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Load California Housing Dataset from sklearn
california_housing = fetch_california_housing()

# Convert the dataset to a DataFrame for easier manipulation
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
data['price'] = california_housing.target  # The target variable is the median house value

# Show the first few rows of the data
print("Data preview:")
print(data.head())

# Step 1: Categorize house prices into bins (for Logistic Regression)
# Define bins and labels for the target variable (house prices)
bins = [0, 1.5, 3.0, np.inf]  # 1.5: Low, 3.0: Medium, Inf: High house values
labels = ['Low', 'Medium', 'High']
data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels)

# Encode the categorical target variable into numerical values
data['price_category'] = data['price_category'].astype('category').cat.codes  # Low=0, Medium=1, High=2

# Step 2: Prepare the features (X) and target variable (y)
X = data.drop(columns=['price', 'price_category'])
y = data['price_category']

# Step 3: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation for model performance
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {cv_scores.mean()}')
