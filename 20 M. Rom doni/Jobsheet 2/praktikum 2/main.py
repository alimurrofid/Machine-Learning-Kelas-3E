import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
#mengimpor dua kelas dari perpustakaan scikit-learn (sklearn):
#LabelEncoder: Digunakan untuk mengubah variabel kategorikal menjadi format numerik, berguna untuk algoritma pembelajaran mesin yang membutuhkan input numerik.
#StandardScaler: Berfungsi untuk menormalkan fitur-fitur numerik dengan cara seperti penskalaan Z-score, menjadikan mean 0 dan deviasi standar 1, membantu menangani masalah penskalaan dalam pembelajaran mesin.

dpath = 'data/Titanic-Dataset-fixed.csv' # dpath adalah variable yang menyimpan data csv
df = pd.read_csv(dpath) #pd.read_csv() untuk membaca berkas CSV yang disimpan dalam variabel dpath


# print(df.head())
# print()
df = df[['Survived', 'Pclass', 'Age', 'Sex', 'Cabin']] # Ini adalah operasi pemilihan kolom di DataFrame df. Dalam tanda kurung ganda, menyebutkan nama kolom-kolom yang ingin dipilih.
# le = LabelEncoder() #  Baris ini membuat objek le dari kelas LabelEncoder. LabelEncoder digunakan untuk mengubah kategori atau label menjadi format numerik, sehingga kolom 'Sex' dan 'Cabin' yang berisi kategori dapat diubah menjadi nilai numerik.
# df['Sex'] = le.fit_transform(df['Sex']) # Baris ini mengubah kolom 'Sex' dalam DataFrame df menjadi nilai numerik. Fungsi fit_transform dari LabelEncoder digunakan untuk mengkodekan nilai-nilai dalam kolom 'Sex' menjadi angka. Misalnya, 'male' menjadi 0 dan 'female' menjadi 1.
# df['Cabin'] = le.fit_transform(df['Cabin']) #Baris ini mengubah kolom 'Cabin' dalam DataFrame df menjadi nilai numerik. Sama seperti sebelumnya, fit_transform digunakan untuk mengkodekan nilai-nilai dalam kolom 'Cabin' menjadi angka.
std = StandardScaler() #Baris ini membuat objek std dari kelas, StandardScaler digunakan untuk menormalkan (menskalakan) fitur-fitur numerik sehingga memiliki mean (rata-rata) 0 dan deviasi standar (standard deviation) 1
df['Age'] = std.fit_transform(df[['Age']]) #Baris ini melakukan skalan pada kolom 'Age' dalam DataFrame df. Untuk melakukan skalan, perlu mengambil kolom 'Age' sebagai array dua dimensi (dengan menggunakan df[['Age']]) karena StandardScaler memerlukan masukan dalam bentuk matriks dua dimensi. Kemudian, perlu menggunakan fit_transform untuk mengubah nilai-nilai dalam kolom 'Age' agar sesuai dengan penskalaan Z-score.
print(df.head())
