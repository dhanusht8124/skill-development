# Step 1: Install required libraries
!pip install pandas scikit-learn

# Step 2: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 3: Load sample SMS spam dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=["label", "message"])
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2)

# Step 5: Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict on custom input
def check_spam(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return "Spam" if pred[0] == 1 else "Not Spam"

# Example
user_input = input("Enter a message to check if it's spam: ")
print("Prediction:", check_spam(user_input))
