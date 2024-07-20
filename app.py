from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the .pkl file
model_filename = 'random_forest_iris_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Map prediction to Iris species
    species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    output = species[prediction[0]]
    
    return render_template('index.html', prediction_text=f'Iris species: {output}')

if __name__ == "__main__":
    app.run(debug=True)
