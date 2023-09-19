# Soal 4
# Lakukan proses stratified split data untuk membuat data latih dan data uji dengan rasio 80:20

# Split data
import pandas as pd
from sklearn.model_selection import train_test_split

# Split data training dan dan testing
dpath = 'Jobsheet 2/dataset/wbc.csv'
df = pd.read_csv(dpath)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df['diagnosis'])

# Print data
print(f'Jumlah label data asli:\n{df.diagnosis.value_counts()}')
print(f'Jumlah label data train:\n{df_train.diagnosis.value_counts()}')
print(f'Jumlah label data test:\n{df_test.diagnosis.value_counts()}')