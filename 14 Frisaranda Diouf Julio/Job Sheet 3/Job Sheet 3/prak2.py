# import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
- numpy, instructs Python to load the NumPy library. NumPy is a powerful library for numerical and mathematical operations in Python.
- matplotlib.pyplot, provides a collection of functions that allow to create and customize various types of plots and charts, such as line plots, bar plots, scatter plots, histograms, and more.
- pandas provides easy-to-use data structures and data analysis tools that make working with structured data, such as tabular data, more efficient and convenient.
'''

# import dataset
dataset = pd.read_csv('Job Sheet 3/dataset/Posisi_gaji.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

'''
- '.iloc[:, 1:2]', this part of the code uses the 'iloc' method to select specific columns from the dataset. in this case, it selects all rows (indicated by ':') and the column at index 1 (the second column, since indexing is 0-based). this typically represents a feature variable or independent variable.
- '.iloc[:, 2], similarly, this part of the code selects all rows and the column at index 2 (the third column) from the dataset. this column typically represents the target variable or dependent variable.
- '.values', this part of the code converts the selected columns into a NumPy array. the resulting array is stored in the variable X or Y.
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1, 1))
y = sc_y.fit_transform(y.reshape(-1, 1))

'''
- 'sc_X = StandardScaler()', this line creates an instance of the 'StandardScaler' class, which will be used to scale your feature variable X.
- 'sc_y = StandardScaler()', this line creates another instance of the 'StandardScaler' class, which will be used to scale your target variable y.
- the 'fit_transform' (in X) method both computes the mean and standard deviation of X and scales its values. the program reshape X to a 2D array with a single feature column using 'reshape(-1, 1)' before applying the scaler because 'StandardScaler' expects a 2D array-like input. After scaling, the standardized values are stored back in the variable X.
- the 'fit_transform' (in y). as with X, the program reshape y to a 2D array with a single column before applying the scaler. the standardized target values are stored back in the variable y.
'''

# Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

'''
- imports the SVR class from scikit-learn's svm module. SVR stands for Support Vector Regression.
- create an instance of the SVR class with the specified kernel function. the 'rbf' kernel stands for Radial Basis Function, which is a commonly used kernel for SVM regression.
- trains the SVR model on your scaled feature variable X and target variable y.
'''

# SVR Result Visualization (high resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Jujur atau tidak (SVR)')
plt.xlabel('Tingkat posisi')
plt.ylabel('Gaji')
plt.show()

'''
- create a range of values for the independent variable X using NumPy's arange function. this range spans from the minimum value of X to the maximum value of X, with a step size of 0.01.
- creates a scatter plot of the original data points. it uses the original X and y values, with the points displayed in red.
- plot the predictions made by the SVR model on the X_grid range of values. the X_grid values are used as input to the 'regressor.predict' method to generate predicted values. the resulting curve is plotted in blue.
- plt.title('Jujur atau tidak (SVR)'), this line sets the title of the plot to 'Jujur atau tidak (SVR)'.
- plt.xlabel('Tingkat posisi'), this line labels the x-axis as 'Tingkat posisi'.
- plt.ylabel('Gaji'), this line labels the y-axis as 'Gaji'.
'''

# Prediction of Results
# Create a 2D array containing the levels of positions to be predicted
tingkat_posisi_prediksi = np.array([[6.5]])
# Feature scalling for the data to be predicted
tingkat_posisi_prediksi = sc_X.transform(tingkat_posisi_prediksi)
# Make predictions using the SVR model
gaji_prediksi = regressor.predict(tingkat_posisi_prediksi)
# Return the prediction results to the original scale
gaji_prediksi = sc_y.inverse_transform(gaji_prediksi.reshape(-1, 1))

# Displays prediction results
print("Prediksi Gaji untuk Tingkat Posisi 6.5:", gaji_prediksi[0])

# SVR Model Evaluation
# Model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

'''
- mean_absolute_error, this metric calculates the mean absolute error (MAE), which is the average of the absolute differences between the predicted values and the actual values.
- mean_squared_error, this metric calculates the mean squared error (MSE), which is the average of the squared differences between the predicted values and the actual values.
- r2_score, this metric calculates the coefficient of determination (R-squared), which quantifies the proportion of the variance in the dependent variable (target) that is explained by the independent variables (features) in the model.
'''

y_actual = y
y_pred = regressor.predict(X)

# Counting MAE
mae = mean_absolute_error(y_actual, y_pred)

# Counting MSE
mse = mean_squared_error(y_actual, y_pred)

# Counting RMSE
rmse = np.sqrt(mse)

# Counting R-squared
r2 = r2_score(y_actual, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared", r2)