# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:07:50 2022

@author: Jacob Davis
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#%% Import Data
def importData():
    dftrain = pd.read_csv('train.csv')
    dftest = pd.read_csv('test.csv')
    
    return dftrain, dftest

#%% Clean Data

def cleanData(df):
    
    df = df.copy(deep=True)
    
    # Drop unnecessary columns
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
        
    return df

#%% Get dataframe with NaNs

def getAllNaNs(df):
    df_nan = df.isna()
    return df_nan 

#%% Dummify categoricals

def dummifyCats(dfTrain, dfTest):
    
    le = preprocessing.LabelEncoder()
    cols = ['Sex', 'Embarked']
    
    for col in cols:
        dfTrain[col] = le.fit_transform(dfTrain[col])
        dfTest[col] = le.transform(dfTest[col])
    
    return dfTrain, dfTest
      

#%% Impute missing values

# Multivaritate Feature Imputation
def imputeMVFI(dfTrain, dfTest):
    
    # Save null values idx
    dfTrainNulls = getAllNaNs(dfTrain)
    dfTestNulls = getAllNaNs(dfTest)
    
    # Save training targets
    trainTarget = dfTrain['Survived']
    
    # Copy dfs
    dfTrain = dfTrain.copy(deep=True)
    dfTest = dfTest.copy(deep=True)
    
    # Remove training targets
    dfTrain.drop('Survived', axis=1, inplace=True)
    
    # Fit imputer
    imp = IterativeImputer(max_iter=10, random_state=0)
    dfTrain[:] = imp.fit_transform(dfTrain)
    dfTest[:] = imp.transform(dfTest)
    
    # Add targets back to train df
    dfTrain['Survived'] = trainTarget
    
    print(dfTest[dfTestNulls].describe())
    return dfTrain, dfTest

# Multivaritate Feature Imputation
def imputeKNN(dfTrain, dfTest):
    
    # Save null values idx
    dfTrainNulls = getAllNaNs(dfTrain)
    dfTestNulls = getAllNaNs(dfTest)
    
    # Save training targets
    trainTarget = dfTrain['Survived']
    
    # Copy dfs
    dfTrain = dfTrain.copy(deep=True)
    dfTest = dfTest.copy(deep=True)
    
    # Remove training targets
    dfTrain.drop('Survived', axis=1, inplace=True)
    
    # Fit imputer
    imp = KNNImputer(n_neighbors=3, weights='uniform')
    dfTrain[:] = imp.fit_transform(dfTrain)
    dfTest[:] = imp.transform(dfTest)
    
    # Print imputed info
    # print("Imputed train data: \n", dfTrain[dfTrainNulls].describe())
    # print("Imputed test data: \n", dfTest[dfTestNulls].describe())
    
    # Add targets back to train df
    dfTrain['Survived'] = trainTarget
    
    
    return dfTrain, dfTest

#%% Split data

def splitData(df, test_size=0.2):
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_val, y_train, y_val


#%% Main

if __name__ == '__main__':

    # Import data
    trainData, testData = importData()

    # Clean data    
    dfTrain = cleanData(trainData)
    dfTest = cleanData(testData)

    # Get null dfs
    dfTrainNulls = getAllNaNs(dfTrain)
    dfTestNulls = getAllNaNs(dfTest)
    
    # Dummify cats (only using label encoder, DID NOT actually dummify vars)
    dfTrain, dfTest = dummifyCats(dfTrain, dfTest)
    
    # Impute missing values
    dfTrainImp, dfTestImp = imputeKNN(dfTrain, dfTest)
    
    # Split train data into train and validation sets
    X_train, X_val, y_train, y_val = splitData(dfTrainImp, 0.2)
    
    ### Train models ###
    lrc = LogisticRegression(random_state=0, max_iter=1000)
    lrc.fit(X_train, y_train)
    
    # Predict
    preds = lrc.predict(X_val)
    print("Validation accuracy: ", accuracy_score(y_val, preds))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    