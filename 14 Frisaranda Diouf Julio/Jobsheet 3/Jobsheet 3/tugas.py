# Tugas 1: Multiple Linear Regression
# 1. Identifikasi variabel-variabel yang akan digunakan sebagai variabel bebas (fitur) dan variabel target (biaya medis personal).
'''
- Variabel Bebas (Fitur)
1) Age (Usia).
2) Sex (Jenis Kelamin)
3) BMI (Body Mass Index)
4) Children (Jumlah Anak)
5) Smoker (Perokok)
6) Region (Wilayah)

- Variabel Target
1) Charges (Biaya Medis Personal)
'''
# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from sklearn import linear_model
import statsmodels.api as sm

# import dataset
dataset = pd.read_csv('Jobsheet 3/dataset/insurance.csv')

# create object from LabelEncoder
le = LabelEncoder()
dataset['sex'] = le.fit_transform(dataset['sex'])
dataset['region'] = le.fit_transform(dataset['region'])
dataset['smoker'] = le.fit_transform(dataset['smoker'])

print(dataset.head())

# data visualization with pairplot
sns.pairplot(dataset, x_vars=['age', 'sex', 'children', 'bmi', 'region', 'smoker'],
             y_vars='charges', height=4, aspect=1, kind='scatter')
plt.show()

# data visualization with heatmap
sns.heatmap(dataset.corr(), cmap="YlGnBu", annot=True)
plt.show()

# create independet variable X and y
x = dataset[['age', 'smoker']]
y = dataset['charges']

# 2. Bagi dataset menjadi data latih (train) dan data uji (test) dengan proporsi yang sesuai.
# splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=100)

# 3. Lakukan feature scaling jika diperlukan.
# Feature scaling tidak perlu dilakukan.

#4. Buat model multiple linear regression menggunakan Scikit-Learn.
# import LinearRegression
from sklearn.linear_model import LinearRegression

# create an object from LinearRegression
mlr = LinearRegression()

# train the linear regression model using data train
mlr.fit(x_train, y_train)


# 5. Latih model pada data latih dan lakukan prediksi pada data uji.
# training model
import statsmodels.api as sm

x_train_sm = sm.add_constant(x_train)
lr = sm.OLS(y_train, x_train_sm).fit()
y_pred = mlr.predict(x)

# 6. Evaluasi model dengan menghitung metrik seperti R-squared, MSE, dan MAE. Tampilkan hasil evaluasi.
# model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_actual = y
y_pred = mlr.predict(x)

# counting MAE
mae = mean_absolute_error(y_actual, y_pred)

# counting MSE
mse = mean_squared_error(y_actual, y_pred)

# counting RMSE
rmse = np.sqrt(mse)

# counting R-squared
r2 = r2_score(y_actual, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)