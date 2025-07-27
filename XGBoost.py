import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

y = data.Price

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
