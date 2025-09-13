# 🏠 House Price Prediction using Machine Learning

This project aims to build a **Machine Learning model** to predict house prices using a dataset containing property details such as the number of rooms, area, location, building type, and more.  
The model helps stakeholders in the **real estate industry** make informed decisions about property valuation.

---

## 📌 Features of the Dataset

- **Id**
- **MSSubClass**
- **MSZoning**
- **LotArea**
- **LotConfig**
- **BldgType**
- **OverallCond**
- **YearBuilt**
- **YearRemodAdd**
- **Exterior1st**
- **BsmtFinSF2**
- **TotalBsmtSF**
- **SalePrice** (target variable)

Dataset file: `HousePricePrediction.csv`

---

## ⚙️ Data Preprocessing

- Identification of categorical, integer, and float variables  
- Exploratory Data Analysis (EDA)  
- Handling missing values  
- Encoding categorical variables using **One-Hot Encoding**  
- Splitting dataset into **training** and **validation sets**  

---

## 🧠 Algorithms Compared

Three algorithms were trained and evaluated:

1. **Support Vector Regressor (SVR)**  
2. **Linear Regression**  
3. **Random Forest Regressor** 🌳 *(best model)*  

---

## 🚀 Model Comparison & Final Execution

### 🔹 Support Vector Regressor (SVR)
```python
from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error

model_svr = svm.SVR()
model_svr.fit(X_train, Y_train)
y_pred_svr = model_svr.predict(X_valid)
print("SVR MAPE:", mean_absolute_percentage_error(Y_valid, y_pred_svr))```


🔹 Linear Regression
```python
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, Y_train)
y_pred_lr = model_lr.predict(X_valid)
print("Linear Regression MAPE:", mean_absolute_percentage_error(Y_valid, y_pred_lr))```



