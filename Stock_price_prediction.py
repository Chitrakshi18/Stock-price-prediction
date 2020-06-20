import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

df= pd.read_csv('prices.csv')
fr_out = 30
df['prediction'] = df[['close']].shift(-fr_out)
X = np.array(df.drop(['prediction'],1))
X = X[:-fr_out]
y = np.array(df['prediction'])
y = y[:-fr_out]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_conf = svr_rbf.score(x_test, y_test)
print("Confidence: ", svm_conf)
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_conf = lr.score(x_test, y_test)
print("Confidence: ", lr_conf)
x_fr = np.array(df.drop(['Prediction'],1))[-fr_out:]
print(x_fr)
lr_pred = lr.predict(x_fr)
print(lr_pred)
svm_pred = svr_rbf.predict(x_fr)
print(svm_pred)
