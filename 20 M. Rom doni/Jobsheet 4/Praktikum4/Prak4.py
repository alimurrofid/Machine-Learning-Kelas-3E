# Load Data

import numpy as np
import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1') # spesifiksi encoding diperlukan karena data tidak menggunakan UTF-8


# Drop 3 kolom terakhir dengan fungsi iloc
df = df.drop(df.iloc[:,2:], axis=1)

# Cek data
# Mengubah Nama Kolom “v1” dan “v2” menjadi “Labels” dan “SMS”
# Data untuk rename kolom v1 dan v2
new_cols = {
    'v1': 'Labels',
    'v2': 'SMS'
}


# Rename nama kolom v1 dan v2
# Drop Kolom Terakhir Yang Tidak Digunakan (NaN)
df = df.rename(columns=new_cols)

# cek data
print(df.head())

# Cek Jumlah Data Per Kelas
print(df['Labels'].value_counts())
print('\n')



# Cek Kelengkapan Data
# Inspeksi Data
print(df.info())
print('\n')

# Cek Statistik Deskriptif
print(df.describe())



# Data untuk label
# Encode Label
new_labels = {
    'spam': 1,
    'ham': 0
}

# Encode label
df['Labels'] = df['Labels'].map(new_labels)

# Cek data
print(df.head())



# Memisahkan Fitur Dengan Label
X = df['SMS'].values
y = df['Labels'].values



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Inisiasi CountVectorizer
bow = CountVectorizer()

# Fitting dan transform X_train dengan CountVectorizer
X_train = bow.fit_transform(X_train)

# Transform X_test
# Mengapa hanya transform? Alasan yang sama dengan kasus pada percobaan ke-3
# Kita tidak menginginkan model mengetahui paramter yang digunakan oleh CountVectorizer untuk fitting data X_train
# Sehingga, data testing dapat tetap menjadi data yang asing bagi model nantinya
X_test = bow.transform(X_test)
# Mencetak Panjang Hasil Ekstraksi Data
print(len(bow.get_feature_names_out()))
print(f'Dimensi data: {X_train.shape}')


# Melakukan Training dan Evaluasi Model

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Inisiasi MultinomialNB
mnb = MultinomialNB()

# Fit model
mnb.fit(X_train, y_train)

# Prediksi dengan data training
y_pred_train = mnb.predict(X_train)

# Evaluasi akurasi data training
acc_train = accuracy_score(y_train, y_pred_train)

# Prediksi dengan data training
y_pred_test = mnb.predict(X_test)

# Evaluasi akurasi data training
acc_test = accuracy_score(y_test, y_pred_test)

# Print hasil evaluasi
print(f'Hasil akurasi data train: {acc_train}')
print(f'Hasil akurasi data test: {acc_test}')
