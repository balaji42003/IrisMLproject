from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
with open('iris.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction
    input_features = np.array([[ sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)[0]

    return f"The predicted Iris class is: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)
