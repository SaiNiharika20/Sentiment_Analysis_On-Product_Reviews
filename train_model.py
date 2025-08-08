import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and clean the data
df = pd.read_csv("reviews.csv")
df = df[df['rating'] != 3]  # Remove neutral ratings

# Create binary labels: 1 for positive (4-5), 0 for negative (1-2)
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['label']

# ⚠️ Use entire dataset for both training and testing (NOT for real-world use)
X_train, X_test, y_train, y_test = X, X, y, y

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
