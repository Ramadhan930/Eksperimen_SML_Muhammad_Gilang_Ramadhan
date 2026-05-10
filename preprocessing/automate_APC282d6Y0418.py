"""
Script otomatisasi preprocessing Loan Prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def load_data(filepath):
    """Membaca CSV mentah."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape}")
    return df

def preprocess(df):
    """Membersihkan dan memproses data."""
    # 1. Isi missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # 2. Encoding
    le = LabelEncoder()
    cat_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 3. Scaling
    scaler = StandardScaler()
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 4. Hapus Loan_ID jika ada
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    print("Preprocessing selesai.")
    return df

def save_data(df, output_path):
    """Simpan data bersih ke CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data disimpan ke {output_path}")

# ----- MAIN -----
if __name__ == "__main__":
    raw_path = "loan_raw/loan.csv"               # input
    processed_path = "loan_preprocessing/loan_clean.csv"  # output
    data = load_data(raw_path)
    data = preprocess(data)
    save_data(data, processed_path)