import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)

df = pd.read_csv("data.csv")

volumes = np.array(df["volume"].tolist()).reshape((-1, 1))
inverses = np.array(df["inverse"].tolist())
pressures = np.array(df["pressure"].tolist())

# X_train, X_test, y_train, y_test = train_test_split(volumes, inverses, test_size=0.1, random_state=36)

# inverse_model = linear_model.LinearRegression().fit(X_train, y_train)
# print(inverse_model.score(X_test, y_test))

poly_features = poly.fit_transform(volumes)
X_train, X_test, y_train, y_test = train_test_split(poly_features, pressures, test_size=0.1, random_state=36)
poly_reg_model = linear_model.LinearRegression()
poly_reg_model.fit(X_train, y_train)
print(poly_reg_model.score(X_test, y_test))