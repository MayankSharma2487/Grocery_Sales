import joblib
import pandas as pd
from catboost import Pool
import numpy as np

# Load models
model_sales = joblib.load("sales_model.pkl")
model_profit = joblib.load("profit_model.pkl")

# Load label encoders
encoders = {}
for col in ['Category', 'Sub Category', 'City', 'Region', 'State', 'Category_Sub']:
    encoders[col] = joblib.load(f"{col}_encoder.pkl")

CATEGORICAL_FEATURES = ['Category', 'Sub Category', 'City', 'Region', 'Category_Sub']

def safe_label_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return -1  # unknown value
    

def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Check if all expected columns are present
    for col in ['Category', 'Sub Category', 'City', 'Region', 'Category_Sub']:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    for col in ['Category', 'Sub Category', 'City', 'Region',  'Category_Sub']:
        df[col] = df[col].apply(lambda x: safe_label_encode(encoders[col], x))
    
    return df

def predict_sales_profit(data):
    df = preprocess_input(data)
    sales_pool = Pool(df, cat_features=CATEGORICAL_FEATURES)
    profit_pool = Pool(df, cat_features=CATEGORICAL_FEATURES)
    
    sales = model_sales.predict(sales_pool)[0]
    profit = model_profit.predict(profit_pool)[0]
    return sales, profit
