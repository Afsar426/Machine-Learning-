#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:32:44 2026

@author: afsarazam
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset =pd.read_csv(r"//Users/afsarazam//Desktop//ML tools& tech//lalita.csv")
x = dataset['YearsExperience']
y = dataset['Salary']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y , test_size=0.20, random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict([[10]])
print(y_pred[10])
