🏠 House Price Prediction using Machine Learning

This project aims to build a Machine Learning model to predict house prices using a dataset containing property details such as the number of rooms, area, location, building type, and more.
The model helps stakeholders in the real estate industry make informed decisions about property valuation.

📌 Features of the Dataset

Id

MSSubClass

MSZoning

LotArea

LotConfig

BldgType

OverallCond

YearBuilt

YearRemodAdd

Exterior1st

BsmtFinSF2

TotalBsmtSF

SalePrice (target variable)

Dataset file: HousePricePrediction.csv

⚙️ Data Preprocessing

Identification of categorical, integer, and float variables

Exploratory Data Analysis (EDA)

Handling missing values

Encoding categorical variables using One-Hot Encoding

Splitting dataset into training and validation sets

🧠 Algorithms Compared

Three algorithms were trained and evaluated:

Support Vector Regressor (SVR)

Linear Regression

Random Forest Regressor 🌳 (best model)

🚀 Model Comparison & Final Execution
Support Vector Regressor (SVR)
'''from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error

model_svr = svm.SVR()
model_svr.fit(X_train, Y_train)
y_pred_svr = model_svr.predict(X_valid)
print("SVR MAPE:", mean_absolute_percentage_error(Y_valid, y_pred_svr))'''

Linear Regression
'''from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)
y_pred_lr = model_lr.predict(X_valid)
print("Linear Regression MAPE:", mean_absolute_percentage_error(Y_valid, y_pred_lr))'''

Random Forest Regressor (Final Chosen Model)
'''from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, Y_train)
y_pred_rf = model_rf.predict(X_valid)
print("Random Forest MAPE:", mean_absolute_percentage_error(Y_valid, y_pred_rf))'''

📊 Results

SVR MAPE: ~0.1870

Linear Regression MAPE: slightly higher error

Random Forest MAPE: ✅ lowest error, best performance

📌 Final Model Selected: Random Forest Regressor

Scatter plot of Actual vs Predicted Prices using Random Forest:

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(Y_valid, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices (Random Forest)")
plt.show()

🔮 Future Work

Perform hyperparameter tuning (GridSearchCV/RandomizedSearchCV) on Random Forest

Try advanced models such as CatBoost or XGBoost

Add feature engineering to improve predictions

Deploy as a web app using Flask or Streamlit