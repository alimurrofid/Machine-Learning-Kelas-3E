# LANGKAH 1 - Load Data
import pandas as pd

data = 'Jobsheet 2/dataset/Titanic-Dataset.csv' # path dataset
df = pd.read_csv(data) # load dataset

df.head()
# df means the DataFrame from Titanic-Dataset and 'head()' method will display the first 5 rows of DataFrame

print(df.head())
# this method will print the 'df.head()' which is the first 5 rows of DataFrame


# LANGKAH 2 - Pengecekan Data
df.info()
# this method will identify missing values, understand the data types, and assess the overall size of DataFrame itself

df.isnull().sum()
# this method will identify columns with missing data in DataFrame

print(df.isnull().sum())
# this method will print the 'df.isnull().sum()'


# LANGKAH 3 - Imputasi
# Age - mean
df['Age'].fillna(value=df['Age'].mean(), inplace=True)

'''
- this code is filling missing values in the 'Age' column.
- '.fillna()' method used to replace missing (NaN) values.
- df['Age'].mean() calculates the mean (average) value of the 'Age' column.
- the mean value is used as the replacement for missing 'Age' values.
- 'inplace=True' means the changes are applied directly to the DataFrame df without the need to assign the result to a new variable.
'''

# Cabin - "DECK"
df['Cabin'].fillna(value="DECK", inplace=True)

'''
- this code is filling missing values in the 'Cabin' column.
- it replaces missing 'Cabin' values with the string "DECK."
'''

# Embarked - modus
df['Embarked'].fillna(value=df['Embarked'].mode, inplace=True)

'''
- this code is filling missing values in the 'Embarked' column.
- '.mode()' is a Pandas method used to find the most frequent value in a Series (the mode).
- df['Embarked'].mode() calculates the mode of the 'Embarked' column.
- the mode value (the most frequent 'Embarked' value) is used as the replacement for missing 'Embarked' values.
'''


# LANGKAH 4 - Validasi Hasil
print(df.head(10))
# print the first 10 rows of DataFrame for validation