# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:59:47 2021

@author: Shivansh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#Dataset taken from Kaggle Heart Disease UCI dataset

df = pd.read_csv(r"heart.csv")

numerical_array = ['age','trestbps','chol','thalach','oldpeak']
categorical_array = ['sex', 'cp', 'fbs', 'restecg','exang',  'slope', 'ca', 'thal']

numerical_columns = df[['age','trestbps','chol','thalach','oldpeak']]
categorical_columns = df[['sex', 'cp', 'fbs', 'restecg','exang',  'slope', 'ca', 'thal']]

sc = StandardScaler()
normalised_columns = sc.fit_transform(numerical_columns)

normalised_df = pd.DataFrame(normalised_columns , columns =  numerical_columns.columns)

ohe = OneHotEncoder()
ohearray = ohe.fit_transform(categorical_columns).toarray()
ohe_df = pd.DataFrame( ohearray , columns = ohe.get_feature_names(categorical_array))

temp_frame = [ normalised_df , ohe_df ]

x = pd.concat(temp_frame , axis = 1 )
y = df["target"]

xTrain , xTest , yTrain , yTest  = train_test_split( x , y )

randomForest = RandomForestClassifier( n_estimators = 100 )
randomForest.fit(xTrain,yTrain)

yPred = randomForest.predict(xTest)
