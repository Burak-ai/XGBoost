import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('melb_data.csv')

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

y = data.Price

X_train, X_valid, y_train, y_valid = train_test_split(X, y)


my_model = XGBRegressor()
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

my_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
                          early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
# set aside some data for calculating the validation scores by setting the eval_set
# use early_stopping_rounds to find the optimal time to stop iterating.
# n_estimators specifies how many times to go through the modeling cycle described above.
# we can multiply the predictions from each model by a small number with learning rate.
# On larger datasets where runtime is a consideration, you can use n_jobs to build your models faster


X = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)



my_model_1 = XGBRegressor(random_state=0)
my_model_1.fit(X_train, y_train)
predictions_1 = my_model_1.predict(X_valid)



