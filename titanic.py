# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:07:50 2022

@author: Jacob Davis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

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


#%% Binning continuous features

#def binFeature(df, featureName):
    
    
    
#%% Label encode categoricals

def labelEncodeCats(dfTrain, dfTest, colsToLabel):
    
    le = preprocessing.LabelEncoder()
        
    for col in colsToLabel:
        dfTrain[col] = le.fit_transform(dfTrain[col])
        dfTest[col] = le.transform(dfTest[col])
    
    return dfTrain, dfTest
      
#%% OneHotEncode categoricals

def oneHotEncodeCats(df, colsToEncode):
    
    for col in colsToEncode:
        # One Hot Encode column
        dumCols = pd.get_dummies(df[col], prefix=col)
        
        # Drop first col of dumCols (not needed)
        dumCols.drop(dumCols.columns[0], axis=1, inplace=True)
        
        # Drop original column in df
        df = df.drop(col, axis=1)
        
        # Add new one hot encoded columns to original df
        df = pd.concat([df, dumCols], axis=1)
        
    return df
        

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

# Random Forest param tester

#def randForParamTest(X_train, X_val, y_train, y_val):
    

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
    
    # Label encode cats
    cols = ['Sex', 'Embarked']
    dfTrain, dfTest = labelEncodeCats(dfTrain, dfTest, cols)
    
    # Impute missing values
    dfTrainImp, dfTestImp = imputeKNN(dfTrain, dfTest)   
    
    # One hot encode cates 
    cols = ['Embarked']
    dfTrain = oneHotEncodeCats(dfTrain, cols)
    
    # Split train data into train and validation sets
    X_train, X_val, y_train, y_val = splitData(dfTrainImp, 0.2)
    
    
    
#%% Train models ###
    
    # Logistic Regression
    lrc = LogisticRegression(random_state=0, max_iter=1000)
    lrc.fit(X_train, y_train)
    predsLRC = lrc.predict(X_val)
    print("LogReg accuracy: ", accuracy_score(y_val, predsLRC))
    
    #%%
    # Random Forest parameter tester
    paramName = 'min_samples_split'
    paramVals = list(range(4,12))
        
    accuracy = np.zeros(len(paramVals))
    rfc = RandomForestClassifier(random_state=0)
    
    for i, val in enumerate(paramVals):
        param_grid = {paramName: [val]}
        grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 
                              cv = 5, n_jobs = -1, verbose = 2)
        grid_search.fit(X_train, y_train)
        preds = grid_search.best_estimator_.predict(X_val)
        accuracy[i] = accuracy_score(y_val, preds)
    
    plt.scatter(paramVals, accuracy)
    
    #%% Rand Forest Grid Search
    
    # Random Forest full grid search
    param_grid = {
            'bootstrap': [True],
            'max_depth': [70, 75, 80],
            'max_features': [2, 4],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 11, 13],
            'n_estimators': [75, 100, 125]
            }
       
    rfc = RandomForestClassifier()
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    predsRFC = best_model.predict(X_val)
    print("RandFor accuracy: ", accuracy_score(y_val, predsRFC))
    
    #%% Ridge Classifier
    
    rc = RidgeClassifierCV()
    rc.fit(X_train, y_train)
    
    predsRC = rc.predict(X_val)
    print("RandFor accuracy: ", accuracy_score(y_val, predsRC))
    
    
#%% Ensemble

    log_clf = LogisticRegression(random_state=0)
    rnd_clf = RandomForestClassifier(random_state=0)
    knn_clf = KNeighborsClassifier()
    svm_clf = SVC(random_state=0)
    
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svm', svm_clf)],
        voting='hard')
    #voting_clf.fit(X_train, y_train)
    
    for clf in (log_clf, rnd_clf, svm_clf, knn_clf, voting_clf):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        print(clf.__class__.__name__, accuracy_score(y_val, pred))
    
#%% Bagging (Bootstrap Aggregation)  
    
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    n_est = list(range(70,126,5))
    max_samp = list(range(175,226,5))
    
    n = 1
    tracker = dict()
    for est in n_est:
        for samp in max_samp:
            
            bag_clf = BaggingClassifier(
                DecisionTreeClassifier(), n_estimators=est, max_samples=samp,
                bootstrap=True, n_jobs=-1, random_state=0)
    
            bag_clf.fit(X_train, y_train)
            pred = bag_clf.predict(X_val)
            
            tracker[(est,samp)] = accuracy_score(y_val, pred)

#%% XGBoost (requires categorical features to be one-hot-encoded)

    xgbc = XGBClassifier()
    xgbc.fit(X_train, y_train)
    
    print(xgbc)
    
    predsXGBC = xgbc.predict(X_val)
    print("XGBC accuracy: ", accuracy_score(y_val, predsXGBC))
    
#%% Lazy Predict

    from lazypredict.Supervised import LazyClassifier
    
    clf = LazyClassifier()
    models,predictions = clf.fit(X_train, X_val, y_train, y_val)
    
    print(models)
    
#%% LGMBClassifier 
    import lightgbm
    lgbm = lightgbm.LGBMClassifier()
    
    lgbm.fit(X_train, y_train)
    predsLGBM = lgbm.predict(X_val)
    print("LGBM accuracy: ", accuracy_score(y_val, predsLGBM))
#%% LGMBClassifier Tuning
    
    
    param_grid = {
            'num_leaves': [2, 5, 10],
            'min_data_in_leaf': [2, 5],
            'max_depth': [1, 2, 3]
            }
       
    lgbm = lightgbm.LGBMClassifier()
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = lgbm, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    predsLGBM = best_model.predict(X_val)
    print("LGBM accuracy: ", accuracy_score(y_val, predsLGBM))
    #%% Boston housing 
    from lazypredict.Supervised import LazyRegressor
    from sklearn import datasets
    from sklearn.utils import shuffle
    import numpy as np
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
    models,predictions = reg.fit(X_train, X_test, y_train, y_test)
    
    
    
    
    
    
    
    
    
    
    
    
    
    