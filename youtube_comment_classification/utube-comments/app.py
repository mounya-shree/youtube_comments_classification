from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('spam_detector_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        output = int(prediction[0])

        return render_template('index.html', prediction_text='Spam' if output == 1 else 'Not Spam')


if __name__ == "__main__":
    app.run(debug=True)
