import pandas as pd

df3 = pd.read_csv('data/Titanic-Dataset-selected.csv')
df3.head()

# Implementasi k-fold cross validation (random) dengan training dan testing saja
from sklearn.model_selection import KFold

# inisiasi obyek kfold
kf = KFold(n_splits=4) # Ini berarti dataset akan dibagi menjadi 4 lipatan atau bagian yang sama besar. Validasi silang adalah teknik yang digunakan untuk mengukur kinerja model dengan cara membagi dataset menjadi beberapa subset (lipatan) dan melatih serta menguji model pada berbagai kombinasi subset tersebut.
print(f'Jumlah fold: {kf.get_n_splits()}') #ini akan mengembalikan nilai 4, yang menunjukkan bahwa dataset akan dibagi menjadi 4 lipatan saat melakukan validasi silang.
print(f'Obyek KFold: {kf}')

# Lakukan splitting dengan KFold
kf_split = kf.split(df3) #digunakan untuk menghasilkan indeks-indeks yang akan digunakan dalam validasi silang K-Fold pada dataset df3
print(f'Jumlah data df: {df3.shape[0]}') #{df3.shape[0]} untuk menyisipkan jumlah baris atau data dalam DataFrame df3. df3.shape adalah atribut yang memberikan dengan dua nilai: jumlah baris (indeks 0) dan jumlah kolom (indeks 1), dan kita menggunakan indeks 0 (df3.shape[0]) untuk mendapatkan jumlah barisnya.

# cek index data tiap fold
#untuk kode program di bawah ini adalah melakukan testing terhadap index dan hasilnya akan di print(tampilkan) dengan menggunakan f=string menyertakan nilai dari train_index dan test_index
# note: train_test dan index_train adalah pendekalarian variable baru dan ini bukan milik kf.split  
for train_index, test_index in kf_split:
    print(f'Index train: {train_index} | Index test: {test_index}')