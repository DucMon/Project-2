import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


train = pd.read_csv("./train.csv", na_values="NAN")
test = pd.read_csv("./test.csv", na_values="NAN")

targets = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)


all_data = pd.concat([train, test])
a = all_data

all_data = pd.get_dummies(all_data)
X = all_data.to_numpy();

X = np.nan_to_num(X)

X.shape

X_train = X[:int(train.shape[0] * 0.8)]
y_train = targets[:int(train.shape[0] * 0.8)]

X_val = X[int(train.shape[0] * 0.8):train.shape[0]]
y_val = targets[int(train.shape[0] * 0.8):]

X_test = X[train.shape[0]:]

print (X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape)
alpha_lasso = [ 5, 10, 12, 20, 50, 100, 200]
rmse = [];
for i in range(7):
    clf = Lasso(alpha = alpha_lasso[i])
    clf.fit(X_train, y_train)
    Y = clf.predict(X_val)
    rmse.append(mean_squared_error(np.log(Y), np.log(y_val))**0.5)

print('Root-Mean-Square-Error: ',rmse);
plt.plot(alpha_lasso,rmse)
plt.ylabel('Root-Mean-Square-Error:')
plt.xlabel('Alpha')
plt.show()

clf = Lasso(alpha = alpha_lasso[rmse.index(min(rmse))])
clf.fit(X_train, y_train)


Y = clf.predict(X_test)
out = pd.DataFrame()
out['Id'] = [i for i in range(train.shape[0]+1,train.shape[0]+test.shape[0]+1)]
out['SalePrice'] = Y
out.to_csv('output_lasso.csv', index=False)