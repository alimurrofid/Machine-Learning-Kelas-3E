import pandas as pd

df2 = pd.read_csv('data/Titanic-Dataset-selected.csv')
df2.head()

# Split data
from sklearn.model_selection import train_test_split # Fungsi ini digunakan untuk membagi dataset menjadi subset pelatihan dan pengujian.

#dua baris kode program di bawah ini sama seperti sebelumnya tapi di sini ditambahkan stratify=df2_unseen['Survived']: Ini adalah argumen tambahan yang menggunakan kolom 'Survived' sebagai kriteria pemisahan (stratifikasi). Dengan menggunakan ini, kita memastikan bahwa proporsi selamat dan tidak selamat dalam data validasi dan pengujian akan mencerminkan proporsi yang sama seperti dalam data asli (df2_unseen).
df2_train, df2_unseen = train_test_split(df2, test_size=0.2, random_state=0, stratify=df2['Survived'])
df2_val, df2_test = train_test_split(df2_unseen, test_size=0.5, random_state=0, stratify=df2_unseen['Survived'])

# Cek masing-masing ukuran data

print(f'Jumlah label data asli:\n{df2.Survived.value_counts()}')
print(f'Jumlah label data train:\n{df2_train.Survived.value_counts()}')
print(f'Jumlah label data val:\n{df2_val.Survived.value_counts()}')
print(f'Jumlah label data test:\n{df2_test.Survived.value_counts()}')