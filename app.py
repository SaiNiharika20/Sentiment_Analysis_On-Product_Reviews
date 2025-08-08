from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    if not review.strip():
        return render_template('index.html', prediction="⚠️ Please enter a review.")
    
    # Vectorize input and predict
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    return render_template('index.html', prediction=sentiment, user_review=review)

if __name__ == '__main__':
    app.run(debug=True)
