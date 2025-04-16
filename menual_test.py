import pickle
import pandas as pd

# Load trained pipeline model
model = pickle.load(open('bangalore_home_prices_model.pkl', 'rb'))

# Input values
location = 'Whitefield'
sqft = 1200
bath = 2
bhk = 3

# Prepare input as DataFrame (column names must match the training DataFrame)
input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

# Predict
prediction = model.predict(input_df)[0]
prediction = round(prediction, 2)

print(f"Predicted price for {location}: ₹ {prediction} Lakhs")
