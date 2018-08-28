# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:25:10 2018

@author: David Timbwa
"""

import numpy as np
from sklearn.externals import joblib

# create an array of values to predict
new_customer = np.array([0,0,100,0,0,780,0]).reshape(1, -1)

# load model from 'clv_predictor.pkl'
model = joblib.load('clv_predictor.pkl')

new_pred = model.predict(new_customer)

print("The CLV of the new customer is: Kshs. ",new_pred[0] - 2322.5150606348125)

