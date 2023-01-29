import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

# ================================================================================== #

# Load data
data_train = pd.read_csv('cars-train.csv')
dataframe_train = pd.DataFrame(data_train)

data_test = pd.read_csv('cars-test.csv')
dataframe_test = pd.DataFrame(data_test)
# Deal with missing values
global final_train
global final_test

# ================================================================================== #


def preprocessing_train(d):
    good_data = d['car-info'].str.split(',', n=2, expand=True)
    d = d.drop('car-info', axis=1)

    # Using DataFrame.insert() to add a column
    d.insert(1, "category", good_data[0], True)
    d.insert(2, "car_name", good_data[1], True)
    d.insert(3, "year_production", good_data[2], True)

    #   drop na values
    new_data = d.dropna(axis=0, how='any')

    # comparing sizes of data frames
    # print("Old data frame length:", len(D), "\nNew data frame length:",
    #       len(new_data), "\nNumber of rows with at least 1 NA value: ",
    #       (len(D) - len(new_data)))

    # fill na values
    # D['volume(cm3)'].fillna(method='ffill', inplace=True)
    # D['drive_unit'].fillna(method='ffill', inplace=True)
    # D['segment'].fillna(method='ffill', inplace=True)

    # Convert "category" to string
    d = new_data.astype({'category': 'string'})
    d['category'] = d['category'].replace("[()[]", "", regex=True)
    d['car_name'] = d['car_name'].replace("[()[]", "", regex=True)
    d['year_production'] = d['year_production'].replace("[()]", "", regex=True)
    d['year_production'] = d['year_production'].replace("]", "", regex=True)
    global final_train
    final_train = d

# ================================================================================== #


def preprocessing_test(d):
    good_data = d['car-info'].str.split(',', n=2, expand=True)
    d = d.drop('car-info', axis=1)
    d['category'] = good_data[0]
    d['car_name'] = good_data[1]
    d['year_production'] = good_data[2]

#   replacing na values
    d['volume(cm3)'].fillna(method='ffill', inplace=True)
    d['drive_unit'].fillna(method='ffill', inplace=True)
    d['segment'].fillna(method='ffill', inplace=True)

    # Convert "category" to string
    d = d.astype({'category': 'string'})
    d['category'] = d['category'].replace("[()[]", "", regex=True)
    d['car_name'] = d['car_name'].replace("[()[]", "", regex=True)
    d['year_production'] = d['year_production'].replace("[()]", "", regex=True)
    d['year_production'] = d['year_production'].replace("]", "", regex=True)
    global final_test
    final_test = d

# ================================================================================== #


preprocessing_train(dataframe_train)
preprocessing_test(dataframe_test)

# ================================================================================== #

columns = ['condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment', 'category', 'car_name']

# ================================================================================== #


def feature_encoder(x, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))
    return x


def featurescaling(fet, a, b):
    fet = np.array(fet)
    normalized_fet = np.zeros((fet.shape[0], fet.shape[1]))
    for i in range(fet.shape[1]):
        normalized_fet[:, i] = ((fet[:, i]-min(fet[:, i]))/(max(fet[:, i])-min(fet[:, i])))*(b-a)+a
    return normalized_fet

# ================================================================================== #


feature_encoder(final_train, columns)
feature_encoder(final_test, columns)

# ================================================================================== #

# Feature Selection
cor = final_train.corr(numeric_only=True, method='pearson')
top_feature = cor.index[abs(cor['price(USD)']) > 0.2]
# Correlation plot
plt.subplots(figsize=(8, 6))
top_corr = final_train[top_feature].corr()
sns.heatmap(top_corr,  cmap="YlGnBu", annot=True)
plt.show()

top_feature = top_feature.delete(-1)
X = final_train[top_feature]
Y = final_train['price(USD)']
# X = featurescaling(X, 0, 1)

# ================================================================================== #

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, random_state=101, test_size=0.22, shuffle=True)
Z = final_test[top_feature]

# ================================================================================== #

# Apply Multiple Linear Regression on the selected features
# never use feature_scaling with Multiple Linear Regression

# cls = linear_model.LinearRegression()
# cls.fit(X_train, y_train)
# prediction = cls.predict(Z)
# df = pd.DataFrame(prediction, columns=['price(USD)'])
# # print(df)
# df.to_csv('Multiple.csv')

# ================================================================================== #

# Apply Polynomial Regression on the selected features

# poly_features = PolynomialFeatures(degree=3)
# poly_model = linear_model.LinearRegression()
# poly_model.fit(X_train, y_train)
# price_Z_predicted = poly_model.predict(Z)
# df = pd.DataFrame(price_Z_predicted, columns=['price(USD)'])
# # print(df)
# df.to_csv('Polynomial.csv')

# ================================================================================== #

# Apply Random Forest Regressor on the selected features

# regressor = RandomForestRegressor(n_estimators=56, random_state=5)
# regressor.fit(X_train, y_train)
# prediction = regressor.predict(Z)
# df = pd.DataFrame(prediction, columns=['price(USD)'])
# # print(df)
# df.to_csv('RFR.csv')

# ================================================================================== #

# Apply Decision Tree Regressor on the selected features
# Do not use feature_scaling with Decision Tree Regressor

# regressor = DecisionTreeRegressor(random_state=2)
# regressor.fit(X_train, y_train)
# prediction = regressor.predict(Z)
# df = pd.DataFrame(prediction, columns=['price(USD)'])
# # print(prediction)
# df.to_csv('DTR.csv')
