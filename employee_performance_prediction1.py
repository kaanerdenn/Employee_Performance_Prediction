import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from lazypredict.Supervised import LazyRegressor

# Set display options for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# Loading data
df_train = pd.read_csv("train_dataset.csv")
df_test = pd.read_csv("test_dataset.csv")
df_train.head(20)
# Function to check DataFrame
def explore_dataframe(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())

# Explore the loaded DataFrame
explore_dataframe(df_train, head=2)

# Replacing the missing values with its mean
df_train["wip"] = df_train["wip"].fillna(df_train["wip"].mean())

# In order to gauge maximum productivity in actual
sns.histplot(df_train['actual_productivity'], kde=False).set(title="Actual productivity gained in general")

# CORRELATION
corrMatrix = df_train.corr()
fig, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax, annot_kws={"size": 8})
plt.show()

# Convert correlation matrix to 1-D Series and sort
sorted_mat = corrMatrix.unstack().sort_values()
pd.set_option('display.max_rows', None)
sorted_mat.tail(100)

# Features and Label Selection
X = df_train.iloc[:, :-1]
y = df_train["actual_productivity"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Algorithms - Famous Regressors
lr = LinearRegression()
dtr = DecisionTreeRegressor()
knr = KNeighborsRegressor()
rfr = RandomForestRegressor()
svr = SVR()

# Fit the models and calculate R-squared values
lr.fit(X_train, y_train)
lr_r2 = lr.score(X_test, y_test)

dtr.fit(X_train, y_train)
dtr_r2 = dtr.score(X_test, y_test)

knr.fit(X_train, y_train)
knr_r2 = knr.score(X_test, y_test)

rfr.fit(X_train, y_train)
rfr_r2 = rfr.score(X_test, y_test)

svr.fit(X_train, y_train)
svr_r2 = svr.score(X_test, y_test)

# Print the R-squared values for each model
print("Linear Regression R-squared:", lr_r2)
print("Decision Tree Regressor R-squared:", dtr_r2)
print("K-Nearest Neighbors Regressor R-squared:", knr_r2)
print("Random Forest Regressor R-squared:", rfr_r2)
print("Support Vector Regressor R-squared:", svr_r2)

# Calculate MSE values for each model
lr_pred = lr.predict(X_test)
knr_pred = knr.predict(X_test)
dtr_pred = dtr.predict(X_test)
rfr_pred = rfr.predict(X_test)
svr_pred = svr.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
knr_mse = mean_squared_error(y_test, knr_pred)
dtr_mse = mean_squared_error(y_test, dtr_pred)
rfr_mse = mean_squared_error(y_test, rfr_pred)
svr_mse = mean_squared_error(y_test, svr_pred)

# Print the MSE values for each model
print("MSE of Linear Regression:", lr_mse)
print("MSE of K-Nearest Neighbors Regressor:", knr_mse)
print("MSE of Decision Tree Regressor:", dtr_mse)
print("MSE of Random Forest Regressor:", rfr_mse)
print("MSE of Support Vector Regressor:", svr_mse)

# Install LazyPredict if not already installed: !pip install lazypredict
# Create a LazyRegressor object
clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit LazyRegressor on the training data and evaluate multiple regression models
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Print the list of models and their corresponding performance metrics
print(models)

# Create the regression models
Gr = GradientBoostingRegressor()
HGr = HistGradientBoostingRegressor()
Ada = AdaBoostRegressor()
knr = KNeighborsRegressor()
xgb_r = xgb.XGBRegressor()
lgbm_r = lgb.LGBMRegressor()

# Fit and evaluate the models
Gr.fit(X, y)
Gr_pred = Gr.predict(X)
MSEGr = mean_squared_error(y, Gr_pred)
R2Gr = r2_score(y, Gr_pred)

HGr.fit(X, y)
HGr_pred = HGr.predict(X)
MSEHGr = mean_squared_error(y, HGr_pred)
R2HGr = r2_score(y, HGr_pred)

Ada.fit(X, y)
Ada_pred = Ada.predict(X)
MSEAda = mean_squared_error(y, Ada_pred)
R2Ada = r2_score(y, Ada_pred)

knr.fit(X, y)
knr_pred = knr.predict(X)
MSEknr = mean_squared_error(y, knr_pred)
R2knr = r2_score(y, knr_pred)

xgb_r.fit(X, y)
xgb_pred = xgb_r.predict(X)
MSE_xgb = mean_squared_error(y, xgb_pred)
R2_xgb = r2_score(y, xgb_pred)

lgbm_r.fit(X, y)
lgbm_pred = lgbm_r.predict(X)
MSE_lgbm = mean_squared_error(y, lgbm_pred)
R2_lgbm = r2_score(y, lgbm_pred)

# Print the results
print("GradientBoostingRegressor MSE:", MSEGr)
print("GradientBoostingRegressor R2:", R2Gr)
print("\nHistGradientBoostingRegressor MSE:", MSEHGr)
print("HistGradientBoostingRegressor R2:", R2HGr)
print("\nAdaBoostRegressor MSE:", MSEAda)
print("AdaBoostRegressor R2:", R2Ada)
print("\nKNeighborsRegressor MSE:", MSEknr)
print("KNeighborsRegressor R2:", R2knr)
print("\nXGBRegressor MSE:", MSE_xgb)
print("XGBRegressor R2:", R2_xgb)
print("\nLGBMRegressor MSE:", MSE_lgbm)
print("LGBMRegressor R2:", R2_lgbm)

# Fitting the XGBoost model
xgb_r.fit(X, y)

# Predict with the XGBoost model
xgb_pred = xgb_r.predict(X)

# Compute MSE
xgb_mse = mean_squared_error(y, xgb_pred)
xgb_r_squared = r2_score(y, xgb_pred)

print("MSE through XGB: %f" % xgb_mse)
print("R-squared through XGB: %f" % xgb_r_squared)

# Create a LightGBM dataset
lgb_train = lgb.Dataset(X, y)
lgb_eval = lgb.Dataset(X, y, reference=lgb_train)

# Define hyperparameters
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learning_rate': 0.05,
    'metric': {'l2', 'l1'},
    'verbose': -1
}

# Train the LightGBM model
model = lgb.train(params, train_set=lgb_train, valid_sets=lgb_eval, early_stopping_rounds=30, verbose_eval=False)

# Make predictions on the test set
y_pred = model.predict(X, num_iteration=model.best_iteration)

# Calculate MSE and R-squared
lgb_mse = mean_squared_error(y, y_pred)
lgb_r_squared = r2_score(y, y_pred)

print("MSE through LightGBM: %f" % lgb_mse)
print("R-squared through LightGBM: %f" % lgb_r_squared)

# Create a DataFrame to store model results
models = pd.DataFrame({
    'Model': ['Random Forest Regressor', 'Ada Boost Regressor', 'Gradient Boosting Regressor',
              'Hist Gradient Boosting Regressor', 'XGBoost', 'LGBM'],
    'R2Score': [R2knr, R2Ada, R2Gr, R2HGr, R2_xgb, lgb_r_squared],
    'Mean Square Error': [MSEknr, MSEAda, MSEGr, MSEHGr, xgb_mse, lgb_mse]
})

# Sort the models by R-squared score in descending order
models.sort_values(by='R2Score', ascending=False, inplace=True)

# Print the sorted DataFrame
print(models)

# Create bar chart using Matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Model names and performance metrics
models = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'Hist Gradient Boosting', 'XGBoost', 'LGBM']
r2_scores = [R2knr, R2Ada, R2Gr, R2HGr, R2_xgb, lgb_r_squared]
mse_scores = [MSEknr, MSEAda, MSEGr, MSEHGr, xgb_mse, lgb_mse]

# Horizontal positions for bar chart
x = np.arange(len(models))

# Bar chart for R2 scores
plt.figure(figsize=(12, 6))
plt.barh(x, r2_scores, color='skyblue', label='R2 Score')
plt.yticks(x, models)
plt.xlabel('R2 Score')
plt.title('Models Comparison - R2 Score')
plt.gca().invert_yaxis()  # Reverse the order of model names
plt.legend()
plt.show()

# Bar chart for MSE scores
plt.figure(figsize=(12, 6))
plt.barh(x, mse_scores, color='lightcoral', label='Mean Squared Error (MSE)')
plt.yticks(x, models)
plt.xlabel('MSE Score')
plt.title('Models Comparison - MSE Score')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
