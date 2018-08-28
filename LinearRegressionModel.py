# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:18:53 2018

@author: David Timbwa
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from time import time
from sklearn.externals import joblib
np.random.seed(7)

raw_data = pd.read_csv("customer_data.csv")



# =============================================================================
## clean data

cleaned_data = raw_data.drop("ID No",axis = 1)
cleaned_data = cleaned_data.drop("index",axis = 1)



cleaned_data = cleaned_data.drop("October",axis = 1)
cleaned_data = cleaned_data.drop("November",axis = 1)
cleaned_data = cleaned_data.drop("December",axis = 1)
cleaned_data = cleaned_data.dropna() # drop rows with NaN values
cleaned_data.corr()['CLV']

print(cleaned_data.head())
features = cleaned_data.drop("CLV",axis = 1)
targets = cleaned_data.CLV


# split data for training and testing in ratio 9:1
features_train,features_test,targets_train,targets_test = train_test_split(features,targets,test_size = .1)

#print("Features-> Training: ",features_train.shape,"\nFeatures--> Testing",features_test.shape)

# Build model on training data
model = LinearRegression()

t0 = time()
model.fit(features_train,targets_train)
print("Training time: ",round(time()-t0,4),"s")

# save model as pickle file
joblib.dump(model, 'clv_predictor.pkl')

# Print coefficients and intercept
print("Coefficients: \n",model.coef_)
print("Intercept: ",model.intercept_)

# Testing :-)
t1 = time()
predictions = model.predict(features_test)
print("Predicting time: ",round(time()-t1,4),"s")

#acc = sklearn.metrics.accuracy_score(predictions,targets_test)
score = sklearn.metrics.r2_score(targets_test,predictions)

print("Score: ",score)





# =============================================================================

