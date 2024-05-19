from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
knn_model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['Overall'], data['Potential'], data['Age'], data['Wage'], data['Height'], data['Weight']]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = knn_model.predict(features_scaled)
    return jsonify({'Value': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
