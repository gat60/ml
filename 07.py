import numpy as np
import pandas as pd
import csv 
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork  
from pgmpy.inference import VariableElimination

heartDisease=pd.read_csv('heart.csv')
heartDisease=heartDisease.replace('?',np.nan)

print('\n Sample instance from the dataset are given below')
print(heartDisease.head())
print('\nAttributes and dytypes')
print(heartDisease.dtypes)

model=DiscreteBayesianNetwork(
                  [('age','heartDisease'),
                   ('sex','heartDisease'),
                   ('exang','heartDisease'),
                   ('cp','heartDisease'),
                   ('heartDisease','restecg'),
                   ('heartDisease','chol')])

print('\nLearning CPD using MaxiumumLikelihood Estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
HeartDiseasetest_infer=VariableElimination(model)

print('\n1.probability of Heart Disease given evidence = restecg:1')
q1=HeartDiseasetest_infer.query(variables=['heartDisease'],evidence={'restecg':1})
print(q1)

print('\n2.probability of Heart Disease given evidence = cp:2')
q2=HeartDiseasetest_infer.query(variables=['heartDisease'],evidence={'cp':2})
print(q2)
