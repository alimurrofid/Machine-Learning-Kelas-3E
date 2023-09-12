import pandas as pd

df = pd.read_csv('data/Titanic-Dataset-selected.csv')
df.head()

# Split data
from sklearn.model_selection import train_test_split #Ini mengimpor fungsi train_test_split dari modul model_selection dalam scikit-learn. Fungsi ini digunakan untuk membagi dataset menjadi subset pelatihan dan pengujian.

df_train, df_unseen = train_test_split(df, test_size=0.2, random_state=0)#Ini adalah pemisahan pertama. Dataset awal df dibagi menjadi dua bagian: df_train: Ini adalah subset data pelatihan. Ini akan digunakan untuk melatih model. df_unseen: Ini adalah subset data yang belum terlihat (unseen data). Biasanya, Anda menggunakan ini untuk menguji model. test_size=0.2 menunjukkan bahwa 20% dari data awal akan menjadi data yang belum terlihat, sedangkan 80% sisanya akan menjadi data pelatihan. random_state=0 digunakan untuk mengatur bibit acak sehingga hasil pengacakan dapat direproduksi jika Anda menjalankan kode ini berkali-kali.

df_val, df_test = train_test_split(df_unseen, test_size=0.5, random_state=0) # Ini adalah pemisahan kedua. Subset data yang belum terlihat (df_unseen) dari langkah sebelumnya dibagi menjadi dua bagian:df_val: Ini adalah subset data validasi. Ini biasanya digunakan untuk mengevaluasi kinerja model saat melakukan penyetelan parameter.df_test: Ini adalah subset data pengujian akhir. Ini digunakan untuk menguji performa model setelah pelatihan dan penyetelan selesai.test_size=0.5 menunjukkan bahwa 50% dari data yang belum terlihat akan menjadi data validasi, dan 50% sisanya akan menjadi data pengujian. random_state=0 di sini juga digunakan untuk memastikan konsistensi hasil pengacakan.

# Cek masing-masing ukuran data
print(f'Jumlah data asli: {df.shape[0]}')
print(f'Jumlah data train: {df_train.shape[0]}')
print(f'Jumlah data val: {df_val.shape[0]}')
print(f'Jumlah data test: {df_test.shape[0]}')

# Cek rasio tiap label
print('=========')
print(f'Jumlah label data asli:\n{df.Survived.value_counts()}')
print(f'Jumlah label data train:\n{df_train.Survived.value_counts()}')
print(f'Jumlah label data val:\n{df_val.Survived.value_counts()}')
print(f'Jumlah label data test:\n{df_test.Survived.value_counts()}')