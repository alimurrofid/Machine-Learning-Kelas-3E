# Soal 3
# Lakukan proses standarisasi pada semua kolom yang memiliki nilai numerik

import pandas as pd
from sklearn.preprocessing import StandardScaler

dpath = 'Jobsheet 2/dataset/wbc.csv'
df = pd.read_csv(dpath)
selected_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
df = df[selected_columns]
std = StandardScaler()
df['area_mean'] = std.fit_transform(df[['area_mean']])
df['perimeter_mean'] = std.fit_transform(df[['perimeter_mean']])
print(df)