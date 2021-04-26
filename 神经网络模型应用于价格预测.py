import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import urllib
from sklearn import datasets
data1=pd.read_excel('数据1.xls', sheet_name='Sheet1') 
SourceData1=data1.drop(["ClPr"], axis=1)
SourceData2=data1["ClPr"].copy()
X_train=SourceData1
y_train=SourceData2
data2=pd.read_excel('数据2.xls', sheet_name='Sheet1') 
SourceData3=data2.drop(["ClPr"], axis=1)
SourceData4=data2["ClPr"].copy()
X_test=SourceData3
y_test=SourceData4
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
#检测   print(np.isnan(data1).any())
#检测   print(np.isnan(data2).any())
mlpr = MLPRegressor(hidden_layer_sizes=(100), activation='relu', solver='adam',alpha=0.0001,max_iter=10,random_state=1)
mlpr.fit(X_train, y_train)
y_pre = mlpr.predict(X_test)
y=mlpr.predict(X_train)
print("mean absolute error(train):", metrics.mean_absolute_error(y_train,y))
print("root mean squared error(train):", (metrics.mean_squared_error(y_train,y)**0.5))
print("mean absolute error(test):", metrics.mean_absolute_error(y_test,y_pre))
print("root mean squared error(test):", (metrics.mean_squared_error(y_test,y_pre)**0.5))
print("在训练集上的R^2:",mlpr.score(X_train,y_train))
print("在测试集上的R^2:",mlpr.score(X_test,y_test))

#模型优化

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPRegressor(hidden_layer_sizes=(100,100), activation='relu', solver='adam',alpha=1,max_iter=10000,random_state=1)
mlp.fit(X_train_scaled, y_train)
y_new_pre = mlp.predict(X_test_scaled)
yy=mlp.predict(X_train_scaled)
print("new mean absolute error(train):", metrics.mean_absolute_error(y_train,yy))
print("new root mean squared error(train):", (metrics.mean_squared_error(y_train,yy)**0.5))
print("new mean absolute error(test):", metrics.mean_absolute_error(y_test,y_new_pre))
print("new root mean squared error(test):", (metrics.mean_squared_error(y_test,y_new_pre)**0.5))
print("优化后在训练集上的R^2:",mlp.score(X_train_scaled,y_train))
print("优化后在测试集上的R^2:",mlp.score(X_test_scaled,y_test))

