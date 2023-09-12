import pandas as pd #Ini adalah kode yang mengimpor library Pandas dan memberikan alias pd

data = 'Titanic-Dataset.csv' # variable data untuk menampung file csv
#df adalah dataframe
df = pd.read_csv(data) #pd.read_csv adalah library dari Pandas untuk membaca berkas CSV yang disimpan dalam variabel data diatas

# print("===Menampilkan baris 5 data teratas===")
# print(df.head()) #menampilkan data menggunakan perintag print dan df.head() untuk menampilkan data 5 baris teratas 
# print()
# print("===Menampilkan info===")
# df.info() #df.info(): Ini adalah perintah yang digunakan untuk menampilkan informasi tentang DataFrame df. Informasi ini mencakup jumlah total baris, jumlah kolom, tipe data setiap kolom, dan jumlah nilai non-null dalam setiap kolom.
# print()
# print("===Menampilkan info yang kosong===")
# print(df.isnull().sum()) #.isnull() digunakan untuk mencari nilai yang hilang (null atau NaN) dalam DataFrame, menghasilkan DataFrame baru dengan nilai True pada lokasi-nilai yang hilang dan False pada lokasi-nilai yang ada. .sum() kemudian menghitung berapa banyak True dalam setiap kolom, yang mewakili jumlah nilai yang hilang (NaN) dalam kolom tersebut.

# Age - mean
df['Age'].fillna(value=df['Age'].mean(), inplace=True) # Ini mengisi nilai yang hilang dalam kolom 'Age' dengan nilai rata-rata usia yang ada dalam kolom tersebut. inplace=True mengubah DataFrame df langsung.

# Cabin - "DECK"
df['Cabin'].fillna(value="DECK", inplace=True) # Ini mengisi nilai yang hilang dalam kolom 'Cabin' dengan string "DECK". Ini digunakan untuk menggantikan nilai yang hilang dengan kategori "DECK" jika informasi deck kabin tidak tersedia.

# Embarked - modus
df['Embarked'].fillna(value=df['Embarked'].mode, inplace=True) #Ini mencoba mengisi nilai yang hilang dalam kolom 'Embarked' dengan mode (nilai paling sering muncul) dari kolom tersebut.

print(df.head(10)) #menampilkan data pada 10 baris teratas