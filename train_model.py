import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset (use local path)
df = pd.read_csv("Bengaluru_House_Data.csv")

# Drop rows with null values in important columns
df = df.dropna(subset=['location', 'size', 'total_sqft', 'bath'])

# Strip spaces from location strings
df['location'] = df['location'].str.strip()

# Save sorted unique location list (for dropdown)
location_list = sorted(df['location'].dropna().unique())
with open('locations.pkl', 'wb') as f:
    pickle.dump(location_list, f)

# Extract number of bedrooms from 'size'
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) and x.split(' ')[0].isdigit() else np.nan)
df = df.dropna(subset=['bhk'])

# Convert 'total_sqft' to float
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft'])

# Final features and target
X = df[['location', 'total_sqft', 'bath', 'bhk']]
y = df['price']

# Pipeline for preprocessing
numeric_features = ['total_sqft', 'bath', 'bhk']
categorical_features = ['location']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
with open("bangalore_home_prices_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

# Save all column names for future reference
location_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['location'])
all_columns = list(location_names) + numeric_features

columns_info = {
    'data_columns': all_columns
}

with open("columns.pkl", "wb") as f:
    pickle.dump(columns_info, f)

print("Column info saved successfully!")
