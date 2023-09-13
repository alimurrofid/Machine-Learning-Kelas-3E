import pandas as pd

# Membaca dataset yang sudah didonwload dan masukan lokasi tempat file tersebut berada 
data = pd.read_csv('data/wbc.csv')

# Menggunakan LabelEncoder untuk encoding kolom "diagnosis"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Menampilkan hasil encoding
print(data['diagnosis'])