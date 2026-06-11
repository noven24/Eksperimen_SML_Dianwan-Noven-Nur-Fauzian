import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    print(f"Loading raw dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return None

    print("Dropping duplicates...")
    df = df.drop_duplicates()
    
    print("Penghapusan Outlier...")
    Q1 = df['charges'].quantile(0.25)
    Q3 = df['charges'].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - (1.5 * IQR)
    batas_atas = Q3 + (1.5 * IQR)
    # Menghapus data outlier
    df = df[(df['charges'] >= batas_bawah) & (df['charges'] <= batas_atas)].copy()
    
    print("Log Transformation...")
    df['charges_log'] = np.log1p(df['charges'])
    
    print("Normalizing numerical columns...")
    numerical_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("Encoding categorical columns...")
    kolom_kategorikal = ['sex', 'smoker', 'region']
    for kolom in kolom_kategorikal:
        le = LabelEncoder()
        df[kolom] = le.fit_transform(df[kolom])
        
    print("Binning charges...")
    bins = [df["charges"].min(),
            df["charges"].quantile(0.25),   
            df["charges"].quantile(0.50),   
            df["charges"].quantile(0.75),   
            df["charges"].max() + 1]
    labels = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

    df['charges_kategori'] = pd.cut(
        df['charges'],
        bins=bins,
        labels=labels,
        right=False
    )
    label_encoder_kategori = LabelEncoder()
    df['charges_kategori'] = label_encoder_kategori.fit_transform(df['charges_kategori'])
        
    df_final = df.copy()
    
    # Drop kolom log dan kategori karena kita tetap menggunakan 'charges' utama untuk modelling
    print("Dropping charges_log and charges_kategori...")
    df_final = df_final.drop(columns=['charges_log', 'charges_kategori'])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Processed dataset shape: {df_final.shape}")
    df_final.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    return df_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing")
    parser.add_argument('--input', type=str, default='namadataset_raw/insurance.csv', help='Path to raw dataset')
    parser.add_argument('--output', type=str, default='preprocessing/namadataset_preprocessing/insurance_preprocessed.csv', help='Path to save processed dataset')
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)