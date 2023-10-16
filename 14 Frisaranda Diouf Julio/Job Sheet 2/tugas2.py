# Soal 2
# Lakukan proses encoding pada kolom "diagnosis"

import pandas as pd

data = pd.read_csv('Job Sheet 2/dataset/wbc.csv')

# Menggunakan LabelEncoder untuk encoding kolom "diagnosis"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Menampilkan hasil encoding
print(data['diagnosis'])