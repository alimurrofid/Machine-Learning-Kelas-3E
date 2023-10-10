# import package
import numpy as np
import pandas as pd

# import library untuk visualisasi

# Training model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



# baca data dari file CSV
data = pd.read_csv('dataset.csv')
# melihat beberapa data awal
# melihat beberapa data awal
print(data.head())
data.head()
#Menampilkan beberapa data awal, ukuran data, informasi data, dan deskripsi statistik data untuk memahami karakteristik data

# mengecek ukuran data
data.shape

# informasi tentang data
data.info()

# deskripsi data
data.describe()

# visualisasi data dengan pairplot
#Menggunakan pairplot untuk menampilkan hubungan antara variabel bebas dan variabel target dalam bentuk scatter plot
sns.pairplot(data, x_vars=['Time on App', 'Time on Website', 'Length of Membership'],
             y_vars='Yearly Amount Spent', size=4, aspect=1, kind='scatter')
plt.show()

# visualisasi korelasi dengan heatmap
#Menggunakan heatmap untuk menampilkan matriks korelasi antara variabel-variabel dalam dataset.
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.show()


# Membuat variabel bebas X dan Y, contoh pengambilan dari analisis korelasi sebelumnya
#Memisahkan variabel bebas (X) dan variabel target (y)
X = data['Length of Membership']
y = data['Yearly Amount Spent']

#Membagi data menjadi data latih (70%) dan data uji (30%) menggunakan train_test_split Lakukan training model regresi linier menggunakan library StatsModels serta menambahkan konstanta (intercept) ke variabel bebas
# Pembagian data latih dan data uji dengan proporsi 7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()
# Visualisasi garis regresi
#Memvisualisasikan garis regresi pada data latih
plt.scatter(X_train, y_train)
plt.plot(X_train, 265.2483 + 66.3015*X_train, 'r')
plt.show()

#Melakukan prediksi nilai y dari data latih dan hitung residual (selisih antara nilai sebenarnya dan nilai prediksi)
# Prediksi nilai y_value dari data x yang telah dilatih
y_train_pred = lr.predict(X_train_sm)

# Menghitung residual
res = (y_train - y_train_pred)

#Memvisualisasikan residual dalam bentuk histogram dan scatter plot untuk mengevaluasi distribusi dan pola error
# Histogram residual
fig = plt.figure()
sns.distplot(res, bins=15)
plt.title('Error Terms', fontsize=15)
plt.xlabel('y_train - y_train_pred', fontsize=15)
plt.show()

# Scatter plot residual
plt.scatter(X_train, res)
plt.show()

#Melakukan prediksi pada data uji
# Prediksi pada data uji
X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)

# Hitung nilai R-squared
#Menghitung nilai R-squared untuk mengukur kinerja model pada data uji
from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_test_pred)

#Memvisualisasikan data uji dan hasil prediksi dalam bentuk scatter plot
# Visualisasi data uji dan hasil prediksi
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()
