from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('bangalore_home_prices_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])
    
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction_text=f'Estimated Price: ₹ {prediction} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)
