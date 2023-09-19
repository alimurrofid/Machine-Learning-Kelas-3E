# import package
import numpy as np
import pandas as pd

'''
- numpy, instructs Python to load the NumPy library. NumPy is a powerful library for numerical and mathematical operations in Python.
- pandas provides easy-to-use data structures and data analysis tools that make working with structured data, such as tabular data, more efficient and convenient.
'''

# read the data from file CSV using pandas
data = pd.read_csv('Jobsheet 3/dataset/dataset.csv')

# display the first 5 rows of the data
data.head()

# check data size
data.shape

# information about data
data.info()

# describe the data
data.describe()

# Data Visualization
# import library for visualization
import matplotlib.pyplot as plt
import seaborn as sns

'''
- matplotlib.pyplot, provides a collection of functions that allow to create and customize various types of plots and charts, such as line plots, bar plots, scatter plots, histograms, and more.
- seaborn is Python library which is specifically for creating visually appealing and informative data visualizations with minimal effort.
'''

# data visualization with pairplot
sns.pairplot(data, x_vars=['Time on App', 'Time on Website', 'Length of Membership'],
             y_vars='Yearly Amount Spent', size=4, aspect=1, kind='scatter')
plt.show()

'''
- pair plot is a grid of scatterplots that allows us to visualize the relationships between multiple pairs of variables in our dataset.
- size, this parameter determines the size of the individual scatterplots in the pair plot. A larger value will result in larger plots.
- aspect, this parameter controls the aspect ratio of the individual scatterplots. It influences how wide or tall each subplot appears.
- kind, this parameter specifies the type of plots to use in the pair plot. In this case, 'scatter' is chosen, which means that scatterplots will be used to visualize the relationships between the variables.
- show, this function will display or show the plot.
'''

# visualization of correlation with heatmap
sns.heatmap(data.corr(), camp="YlGnBu", annot=True)
plt.show()

'''
- heatmap is a graphical representation of data where individual values are represented as colors.
- data.corr(), this part calculates the correlation matrix of our dataset. the corr() method computes the pairwise correlations between numerical columns in our DataFrame. The resulting correlation matrix shows how each variable relates to every other variable, with values ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- cmap="YlGnBu", this parameter specifies the colormap to be used for coloring the heatmap. The "YlGnBu" colormap is a yellow-green-blue colormap, which is often used for visualizing data with positive correlations.
- annot=True, this parameter controls whether or not the values in the cells of the heatmap are annotated (displayed) within the cells. When set to True, the values will be displayed; when set to False, they won't be.
'''

# Regresi Linear
# create independent variables X and Y, examples taken from previous correlation analysis
X = data['Length of Membership']
y = data['Yearly Amount Spent']

'''
set X as 'Length of Membership' and Y as 'Yearly Amount Spent'
'''

# dividing training data and test data with a proportion of 7:3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

'''
- model_selection, this function is commonly used for splitting a dataset into training and testing subsets.
- 'train_size=0.7' and 'test_size=0.3', means that 70% of the data will be used for training (X_train and y_train), and 30% will be used for testing (X_test and y_test).
- 'random_state=100', this parameter sets a seed for the random number generator used during the split. Setting it to a specific value (e.g., 100) ensures that the split is reproducible. If you omit this parameter, the split will be different each time you run the code.
'''

# training model
import statsmodels.api as sm

'''
'statsmodels' is a Python library that provides classes and functions for the estimation of various statistical models and performing statistical tests. It is particularly useful for statistical analysis, hypothesis testing, and regression modeling.
'''

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

'''
- the 'add_constant' function from 'statsmodels' adds a column of 1s to the beginning of our feature matrix, representing the intercept term.
- 'sm.OLS(y_train, X_train_sm)', this part creates an OLS regression model object with 'y_train' as the dependent variable and 'X_train_sm' as the independent variables (features).
- '.fit()', this method fits the OLS model to our training data, estimating the coefficients (parameters) of the linear regression model.
'''

# regression line visualization
plt.scatter(X_train, y_train)
plt.plot(X_train, 265.2483 + 66.3015*X_train, 'r')
plt.show()

'''
- this line creates a scatter plot of our training data. it uses X_train as the x-axis values and y_train as the y-axis values.
- this line plots a red ('r') regression line on top of the scatter plot.
- '265.2483' is the intercept term (constant).
- '66.3015' is the coefficient associated with the independent variable X_train.
- 'X_train' represents the values of the independent variable(s) from our training data.
- 'plt.show()', this command displays the plot on the screen.
'''

# Analisis Residual
# predict the y_value from the x_data that has been trained
y_train_pred = lr.predict(X_train_sm)

'''
- this part uses the 'predict' method of the 'lr' model to generate predictions for the target variable '(y_train)' based on the provided feature matrix. the predictions are stored in the variable 'y_train_pred'.
'''

# calculate residuals
res = (y_train - y_train_pred)

'''
- this code calculates the residuals of our linear regression model on the training data. residuals represent the differences between the actual target values '(y_train)' and the predicted values '(y_train_pred)' generated by the model.
'''

# residual histogram
fig = plt.figure()
sns.distplot(res, bins=15)
plt.title('Error Terms', fontsize=15)
plt.xlabel('y_train - y_train_pred', fontsize=15)
plt.show()

'''
- 'fig = plt.figure()', this line initializes a new figure for plotting.
- 'sns.distplot(res, bins=15)', this line uses Seaborn's 'distplot' function to create a histogram and density plot of the residuals (res).
- 'res', this is the array of residuals that were calculated earlier.
- 'bins=15', this parameter specifies the number of bins or intervals to use for the histogram. in this case, it's set to 15, meaning the histogram will have 15 bins.
- 'plt.title('Error Terms', fontsize=15)', this line sets the title of the plot to 'Error Terms' with a font size of 15.
- 'plt.xlabel('y_train - y_train_pred', fontsize=15)', this line labels the x-axis as 'y_train - y_train_pred' with a font size of 15.
- 'plt.show()', this command displays the plot.
'''

# residual scatter plot
plt.scatter(X_train, res)
plt.show()

'''
'plt.scatter(X_train, res)', this line generates a scatter plot. it uses 'X_train' as the x-axis values (independent variable(s)) and 'res' as the y-axis values (residuals). each point on the plot represents a data point from our training dataset.
'''

# predictions on test data
X_test_sm = sm.add_constant(X_test)
y_test_pred = lr.predict(X_test_sm)

'''
this line code is similar to what I did for the training data.
'''

# calculate the R-squared value
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_test_pred)

'''
- 'from sklearn.metrics import r2_score', this line imports the 'r2_score' function, which is used to calculate the R-squared value, a common metric for evaluating the performance of regression models.
- 'y_test', this variable contains the actual target values (ground truth) for our testing data.
- 'y_test_pred', this variable contains the predicted values for the target variable generated by our linear regression model when applied to the testing data.
- 'r_squared = r2_score(y_test, y_test_pred)', this line calculates the R-squared value by comparing the actual target values '(y_test)' with the predicted values '(y_test_pred)' and assigning the result to the variable 'r_squared'.
'''

# visualization of test data and prediction results
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()

'''
- 'plt.scatter(X_test, y_test)', this line generates a scatter plot of our testing data. it uses 'X_test' as the x-axis values (independent variable(s)) and 'y_test' as the y-axis values (actual target values). each point on the plot represents a data point from our testing dataset.
- 'plt.plot(X_test, y_test_pred, 'r')', this line plots a red ('r') regression line on top of the scatter plot. the regression line is based on the predicted values '(y_test_pred)' generated by our linear regression model. it shows how the model's predictions compare to the actual data points.
'''