from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and columns
model = pickle.load(open('bangalore_home_prices_model.pkl', 'rb'))

# Load the data or column names to extract locations
data_columns = pickle.load(open("columns.pkl", "rb"))  # assuming you saved this when training
locations = data_columns['data_columns'][3:]  # first 3: sqft, bath, bhk; rest: location one-hot columns

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])

        input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        # Choose image based on predicted price
        if prediction < 50:
            image_file = 'low_price.jpg'
        elif prediction <= 100:
            image_file = 'mid_price.jpeg'
        else:
            image_file = 'high_price.jpeg'

        return render_template('index.html',
                               prediction_text=f'Estimated Price: ₹ {prediction} Lakhs',
                               image_file=image_file,
                               locations=locations)

    except Exception as e:
        return render_template('index.html', error_message=f"Error: {e}", locations=locations)

if __name__ == "__main__":
    app.run(debug=True)
