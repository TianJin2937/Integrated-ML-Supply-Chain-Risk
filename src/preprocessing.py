"""Data preprocessing for supply chain risk prediction."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_and_preprocess(filepath):
    """Load DataCo dataset and apply preprocessing pipeline."""
    df = pd.read_csv(filepath, encoding='latin-1')

    # Select relevant features
    feature_cols = [
        'Days for shipping (real)', 'Days for shipment (scheduled)',
        'Benefit per order', 'Sales per customer', 'Order Item Discount',
        'Order Item Profit Ratio', 'Order Item Quantity', 'Sales',
        'Order Item Total', 'Product Price'
    ]
    categorical_cols = ['Shipping Mode', 'Category Name', 'Market', 'Order Region']

    # Handle missing values
    for col in feature_cols:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Create risk label: Late delivery = 1, On-time = 0
    df['risk_label'] = (df['Late_delivery_risk']).astype(int)

    # Outlier detection using IQR
    for col in feature_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

    # Normalize numerical features
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df[feature_cols].values)

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_data = df[categorical_cols].astype(str)
    X_cat = encoder.fit_transform(cat_data)

    X = np.hstack([X_num, X_cat])
    y = df['risk_label'].values

    return X, y, scaler, encoder
