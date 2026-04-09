#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 22:33:12 2026

@author: afshu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv(r"/Users/afsarazam/Desktop/ML tools& tech/salary_data.csv")
dataset2 =pd.read_csv(r"/Users/afsarazam/Desktop/ML tools& tech/af.csv")
x= dataset.iloc[:, :-1].values
y=dataset.iloc[:,1].values
#we can train data in 80% and test data 20% 


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.40, random_state=1)# if change the by deflut paremeter thats called hyperparemeter tunning


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict((pd.DataFrame(dataset2)))
print(y_pred)
