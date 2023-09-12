import pandas as pd
from sklearn.preprocessing import StandardScaler

dpath = 'data/wbc.csv'
df = pd.read_csv(dpath)
selected_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
df = df[selected_columns]
std = StandardScaler()
df['area_mean'] = std.fit_transform(df[['area_mean']])
df['perimeter_mean'] = std.fit_transform(df[['perimeter_mean']])
df.head()
print(df)