import nbformat as nbf
import os
import subprocess

nb = nbf.v4.new_notebook()

# 1. Perkenalan Dataset
nb.cells.append(nbf.v4.new_markdown_cell("""# Eksperimen MSML: Data Preprocessing

## 1. Perkenalan Dataset

Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:
1. **Sumber Dataset:**
   Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (Kaggle, UCI ML Repository, Open Data) atau data primer yang Anda kumpulkan sendiri.

*Dataset yang digunakan pada eksperimen ini adalah dataset penjualan (sales data) yang diambil dari sumber internal perusahaan untuk memprediksi nilai (target) di masa depan berdasarkan tren historis (time-series).*
"""))

# 2. Import Library
nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Import Library

Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning.
"""))
nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
import numpy as np"""))

# 3. Memuat Dataset
nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Memuat Dataset

Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.
"""))
nb.cells.append(nbf.v4.new_code_cell("""df = pd.read_csv('../namadataset_raw/dataset.csv')
df.head()"""))

# 4. EDA
nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Exploratory Data Analysis (EDA)

Pada tahap ini, Anda akan melakukan **Exploratory Data Analysis (EDA)** untuk memahami karakteristik dataset. Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan.
"""))
nb.cells.append(nbf.v4.new_code_cell("""print("Info Dataset:")
print(df.info())
print("\\nDeskripsi Statistik:")
print(df.describe())"""))

# 5. Data Preprocessing
nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Data Preprocessing

Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.
Berikut adalah tahapan-tahapan yang bisa dilakukan:
1. Menghapus atau Menangani Data Kosong (Missing Values)
2. Menghapus Data Duplikat
3. Normalisasi atau Standarisasi Fitur
4. Deteksi dan Penanganan Outlier
5. Encoding Data Kategorikal
6. Binning (Pengelompokan Data)
"""))
nb.cells.append(nbf.v4.new_code_cell("""# 1. Konversi format tanggal
df['ds'] = pd.to_datetime(df['Time Date'].astype(str).str.zfill(8), format='%d%m%Y', errors='coerce')

# 2. Filtering data untuk satu produk dan toko spesifik
df_prophet = df.loc[(df['Product'] == 2667437) & (df['Store'] == 'QLD_CW_ST0203')].copy()

# 3. Rename target column to 'y'
df_prophet = df_prophet.rename(columns={'Value': 'y'})
df_prophet = df_prophet[['ds', 'y']].sort_values('ds')

# 4. Menangani Data Kosong dan Duplikat
df_prophet = df_prophet.dropna(subset=['ds'])
df_prophet = df_prophet.drop_duplicates()

print(f"Processed dataset shape: {df_prophet.shape}")
df_prophet.head()"""))

notebook_path = "Eksperimen_Dianwan-Noven-Nur-Fauzian.ipynb"
nbf.write(nb, notebook_path)

# Jalankan notebook menggunakan papermill atau nbconvert
print("Menjalankan notebook untuk menghasilkan output...")
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", notebook_path], check=True)
print("Notebook berhasil diperbarui dan dijalankan!")
