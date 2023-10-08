# TUGAS 1
print("Tugas 1")

# # Untuk data manipulation
import pandas as pd

# Untuk data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Untuk data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

# Untuk SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('voice.csv')
print(df.head())

print()

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
print(df.head())


print()
print(df.isnull().sum())

print()
print(df.describe())

corr = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='magma')
plt.title('Correlation Heatmap', fontsize=20)
plt.show()

print()
# Menampilkan nilai korelasi antara fitur-fitur dengan label
corr_rank = corr.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
print(corr_rank[corr_rank['level_0'] == 'label'])

# Memisahkan fitur (X) dan label (y)
X = df.drop('label', axis=1)  
y = df['label']  

# Memisahkan data menjadi data training dan testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

#  Melatih model Support Vector Machine (SVM) dengan kernel linear
svm_model = SVC(kernel='linear') 
svm_model.fit(X_train, y_train)

# Memprediksi label pada data testing
y_pred = svm_model.predict(X_test)

# Evaluasi model dengan menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

 # TUGAS 2
print("Tugas 2")
df2 = pd.read_csv('spam.csv', encoding='latin-1') # Membaca dataset CSV 
print(df2.head())

print()

# Menghapus tiga kolom yang tidak bernama (Unnamed: 2, Unnamed: 3, Unnamed: 4)
df2 = df2.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
print(df2.head())

print()

# Mengganti nama kolom 'v1', 'v2' menjadi 'labels' dan  menjadi 'SMS'
new_cols = {
	'v1': 'labels',
	'v2': 'SMS'
} 

df2 = df2.rename(columns=new_cols)
print(df2.head())

print()
# Menghitung jumlah masing-masing label (spam dan ham)
print(df2.value_counts('labels'))

print()
# Menampilkan informasi mengenai dataset, seperti jumlah data dan tipe data
print(df2.info())

print()
# Menampilkan deskripsi dari dataset
print(df2.describe())

print()
# Mengganti label 'ham' menjadi 0 dan 'spam' menjadi 1
new_label = {
	'ham': 0,
	'spam': 1
}

df2['labels'] = df2['labels'].map(new_label)
print(df2.head())

# Mengambil nilai fitur (SMS) dan label (labels) dari dataset
X = df2['SMS'].values
y = df2['labels'].values

# Inisiasi CountVectorizer dengan mengaktifkan stop words dalam bahasa Inggris
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X) # Transformasi teks menjadi vektor numerik
print(vectorizer.get_feature_names_out())

print()

# Membagi data menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mnb = MultinomialNB() # Inisiasi model klasifikasi Multinomial Naive Bayes

mnb.fit(X_train, y_train) # Melatih model dengan data training

y_pred_train = mnb.predict(X_train) # Prediksi label dengan data training
acc_train = accuracy_score(y_train, y_pred_train) # Evaluasi akurasi pada data training
y_pred_test = mnb.predict(X_test) # Prediksi label dengan data testing
acc_test = accuracy_score(y_test, y_pred_test)# Evaluasi akurasi pada data testing


# Menampilkan hasil evaluasi akurasi
print('Accuracy on train data: ', acc_train)
print('Accuracy on test data: ', acc_test)


# TUGAS 3

print()
print("Tugas 3")

# ini digunakan untuk membuat objek TfidfVectorizer yang akan digunakan dalam pemrosesan teks
# Inisiasi TfidfVectorizer dengan mengaktifkan stop words dalam bahasa Inggris
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Transformasi teks menjadi vektor numerik berdasarkan nilai TF-IDF
X = df2['SMS'].values
y = df2['labels'].values

X = tfidf_vectorizer.fit_transform(X)

# Menampilkan daftar fitur yang diekstrak oleh TfidfVectorizer
print(tfidf_vectorizer.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# prediksi dan evaluasi akurasi
mnb = MultinomialNB()

mnb.fit(X_train, y_train)

y_pred_train = mnb.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)

y_pred_test = mnb.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)

print('Accuracy on train data: ', acc_train)
print('Accuracy on test data: ', acc_test)

# Kesimpulan yang dapat diambil dari analisis ini adalah bahwa untuk dataset "spam.csv," CountVectorizer merupakan pilihan fitur yang lebih baik. CountVectorizer mengaktifkan penggunaan "stop_words" untuk mengabaikan kata-kata penghubung dalam bahasa Inggris saat menghitung frekuensi kata-kata dalam dokumen. Hasilnya, model klasifikasi Multinomial Naive Bayes yang menggunakan CountVectorizer memberikan akurasi yang lebih tinggi pada kedua dataset pelatihan dan pengujian. Dalam konteks ini, CountVectorizer menunjukkan kinerja yang lebih baik dalam mengklasifikasikan data spam dan non-spam. Ini menandakan bahwa pengabaian kata-kata penghubung dalam bahasa Inggris membantu model untuk lebih fokus pada kata-kata kunci yang mungkin menjadi penanda untuk mengidentifikasi pesan spam. Oleh karena itu, pemilihan CountVectorizer sebagai fitur lebih disarankan dalam kasus ini dibandingkan dengan TfidfVectorizer.