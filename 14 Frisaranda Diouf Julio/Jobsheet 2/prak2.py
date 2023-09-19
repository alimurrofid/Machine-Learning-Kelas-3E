# LANGKAH 0 - Load Library
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

'''
- 'LabelEncoder' to convert categorical target labels into numerical format.
- 'StandardScaler' to scale your input features before feeding them into a machine learning model.
'''


# LANGKAH 1 - Load Data
dpath = 'Jobsheet 2/dataset/Titanic-Dataset-fixed.csv'
df = pd.read_csv(dpath)
print(df.head())


# LANGKAH 2 - Slice Data
df = df[['Survived', 'Pclass', 'Age', 'Sex', 'Cabin']]
print(df.head())

'''
this code is used to select the variables which we consider as features.
'''


# LANGKAH 3 - Encoding
le = LabelEncoder() # membuat objek dari LabelEncoder
df['Sex'] = le.fit_transform(df['Sex']) # proses encoding
df['Cabin'] = le.fit_transform(df['Cabin']) # proses encoding

'''
this code will replacing the original categorical values with numerical labels.
'''


# LANGKAH 4 - Verifikasi Hasil
print(df.head())


# LANGKAH 5 - Standarisasi
std = StandardScaler()
df['Age'] = std.fit_transform(df[['Age']])

'''
- the fit method calculates the mean and standard deviation of the 'Age' column.
- the transform method then standardizes the 'Age' column by subtracting the mean and dividing by the standard deviation.
- the result is assigned back to the 'Age' column in your DataFrame, effectively replacing the original 'Age' values with their standardized counterparts.
'''


# LANGKAH 6 - Verifikasi Hasil Standarisasi
print(df.head())