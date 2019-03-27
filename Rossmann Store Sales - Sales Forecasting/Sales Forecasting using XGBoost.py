# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:58:28 2018

@author: Vishal
"""
#-------------------------------Importing Libraries----------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV , KFold
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, \
RandomForestRegressor, AdaBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV

#-------------------------------Creating RMSPE functions-----------------------
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

#--------------------------------Loading input files---------------------------
store = pd.read_csv("store.csv")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#--------------------------------Cleaning store data---------------------------
store.isna().sum()
store.fillna(0, inplace = True)
store.dtypes
mapping = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
store['StoreType']=store['StoreType'].map(mapping)
store['Assortment']=store['Assortment'].map(mapping)
#store.iloc[:,-1].value_counts()

#--------------------------------Cleaning train data---------------------------
train.isna().sum()
train.dtypes
train['Date'] = train['Date'].astype('datetime64')
#train['StateHoliday'].value_counts()
train['StateHoliday']=train['StateHoliday'].replace(mapping)

#---------------------------Feature Extraction on train data-------------------
train_store = pd.merge(train, store, on = 'Store', how = 'left')
train_store.isna().sum()
train_store['Year'] = pd.to_datetime(train_store['Date']).dt.year
train_store['Month'] = pd.to_datetime(train_store['Date']).dt.month
train_store['Week'] = pd.to_datetime(train_store['Date']).dt.week
train_store['Weekend'] = train_store['DayOfWeek'].apply(lambda x: 1 if x > 5 \
           else 0)
train_store['Day'] = pd.to_datetime(train_store['Date']).dt.day
month_str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
train_store['Month_str'] = train_store['Month'].map(month_str)
train_store['MonthsSinceComp'] = (train_store['Year']- \
           train_store['CompetitionOpenSinceYear'])*12 + \
           (train_store['Month']-train_store['CompetitionOpenSinceMonth'])
train_store['MonthsSincePromo2'] = (train_store['Year']-
           train_store['Promo2SinceYear'])*12 + (train_store['Month']- \
                      train_store['Promo2SinceWeek']) / 4           
train_store['MonthsSinceComp']=train_store['MonthsSinceComp'].apply(lambda x: \
           x if x>=0 else 0)
train_store['MonthsSincePromo2']=train_store['MonthsSincePromo2']. \
apply(lambda x:x if x>=0 else 0)
train_store['MonthsSinceComp']=np.where(train_store['CompetitionOpenSinceYear']\
           ==0,
           0, train_store['MonthsSinceComp'])
train_store['MonthsSincePromo2']=np.where(train_store['Promo2SinceYear']==0,\
           0, train_store['MonthsSincePromo2'])
train_store['PromoMonth']=train_store.apply(lambda x: 1 if x['Month_str'] in \
           str(x['PromoInterval']) else 0, axis = 1)

#-----------------------------Cleaning test data-------------------------------
test.isna().sum()
test.dtypes
test.fillna(1, inplace=True)
test['Date'] = test['Date'].astype('datetime64')
test['StateHoliday'].value_counts()
test['StateHoliday']=test['StateHoliday'].replace(mapping)

#-----------------------Feature extraction on test data------------------------
test_store = pd.merge(test, store, on = 'Store', how = 'left')
test_store['Year'] = pd.to_datetime(test_store['Date']).dt.year
test_store['Month'] = pd.to_datetime(test_store['Date']).dt.month
test_store['Week'] = pd.to_datetime(test_store['Date']).dt.week
test_store['Weekend'] = test_store['DayOfWeek'].apply(lambda x: 1 if x > 5 \
          else 0)
test_store['Day'] = pd.to_datetime(test_store['Date']).dt.day
month_str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
             7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
test_store['Month_str'] = test_store['Month'].map(month_str)
test_store['MonthsSinceComp'] = (test_store['Year']- \
           test_store['CompetitionOpenSinceYear'])*12 + \
           (test_store['Month']-test_store['CompetitionOpenSinceMonth'])
test_store['MonthsSincePromo2'] = (test_store['Year']-
           test_store['Promo2SinceYear'])*12 + (test_store['Month']- \
                      test_store['Promo2SinceWeek']) / 4
test_store['MonthsSinceComp']=test_store['MonthsSinceComp'].apply(lambda x: \
           x if x>=0 else 0)
test_store['MonthsSincePromo2']=test_store['MonthsSincePromo2'].apply(lambda \
          x:x if x>=0 else 0)
test_store['MonthsSinceComp']=np.where(test_store['CompetitionOpenSinceYear']==\
          0, 0, test_store['MonthsSinceComp'])
test_store['MonthsSincePromo2']=np.where(test_store['Promo2SinceYear']==0,\
           0, test_store['MonthsSincePromo2'])
test_store['PromoMonth']=test_store.apply(lambda x: 1 if x['Month_str'] in \
           str(x['PromoInterval']) else 0, axis = 1)

#---------------------------Taking only Open store positive sales--------------
train_store = train_store[train_store['Open']==1]
train_store = train_store[train_store['Sales']>0]

#------------------Creating X and Y datasets for model training----------------
X = train_store[['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',\
                'StoreType', 'Assortment', 'CompetitionDistance',
                'CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2', \
                'Promo2SinceWeek','Promo2SinceYear','Month', 'Week', 'Weekend',\
                'Day', 'MonthsSinceComp', 'MonthsSincePromo2', 'PromoMonth']]
Y = np.log1p(train_store[['Sales']])

#----------------------------Train and test split------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, \
                                                  random_state=7)

#----------------------------Top features--------------------------------------
feat = RandomForestRegressor(n_estimators=10, random_state=1)
feat.fit(X_train, y_train)

feature_importance = pd.DataFrame(feat.feature_importances_,
                                   index = X.columns,
                                columns=['importance']).sort_values('importance',
                                ascending = False)

top_features = feature_importance.nlargest(10,"importance")

#---------------------Creating test dataset similar to train-------------------
X_test = test_store[X_train.columns]

#--------------------------Model training--------------------------------------
#--------------------------Normalization---------------------------------------
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_train)
X_train_scaled = pd.DataFrame(min_max_scaler.transform(X_train))
X_val_scaled = pd.DataFrame(min_max_scaler.transform(X_val))

#--------------------------Elastic Net Regression------------------------------
reg = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1],
                   alphas=[0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.],
                   cv = 5)

reg = reg.fit(X_train_scaled, y_train)

y_pred = reg.predict(X_val_scaled)
y_pred = np.where(y_pred<0, 0, y_pred)

error_cal = pd.concat([y_val, pd.DataFrame(y_pred)], axis = 1)
error_cal.columns = ['Actual', 'Predicted']
error = rmspe(np.expm1(error_cal['Actual']), np.expm1(error_cal['Predicted']))
print('RMSPE: {:.4f}'.format(error))
#0.2417

#--------------------------Random Forest---------------------------------------
params = {'n_estimators': list(range(200,1200, 200)),
          'criterion':['mse'],
          'max_depth': [5,8,10,15],
          'min_samples_split': [5, 10, 15],
          'bootstrap': [True,False]}

reg = RandomForestRegressor()
reg.fit(X_train, y_train)
reg = reg.best_estimator_

y_pred = reg.predict(X_val, y_val)
y_pred = np.where(y_pred<0, 0, y_pred)
    
error_cal = pd.concat([y_val, pd.DataFrame(y_pred)], axis = 1)
error_cal.columns = ['Actual', 'Predicted']
error = rmspe(np.expm1(error_cal['Actual']), np.expm1(error_cal['Predicted']))
print('RMSPE: {:.4f}'.format(error))
#0.1608

#----------------------------------XGB-----------------------------------------
params = {"objective": "reg:linear", # for linear regression
          "booster" : ["gbtree", "gblinear"],   # use tree based models 
          "eta": [0.1, 0.3],  # learning rate
          "max_depth": list(range(5, 15, 2)),   # maximum depth of a tree
          "subsample": 0.9,    # Subsample ratio of the training instances
          "colsample_bytree": 0.7, # Subsample ratio of columns
          "seed": 10,   # Random number seed
          #"n_estimators" : [50],
          "verbose":True,
          "n_estimators" : list(range(200, 1200, 200))
          }

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

model = GridSearchCV(XGBRegressor(), params, cv = 5)
model.fit(X,Y)
model = model.best_estimator_

ypred = model.predict(xgb.DMatrix(X_val))
ypred = np.where(ypred<0, 0, ypred)

error_cal = pd.concat([y_val, pd.DataFrame(ypred)], axis = 1)
error_cal.columns = ['Actual', 'Predicted']
error = rmspe(np.expm1(error_cal['Actual']), np.expm1(error_cal['Predicted']))
print('RMSPE: {:.4f}'.format(error))
#0.1194

#----------------------------Gradient Boosting Model---------------------------
GBR = GradientBoostingRegressor()

# Parameter grid
parameter_grid = {'n_estimators' : list(range(30, 500, 50)),
                  'learning_rate': [0.1, 0.3],
                  'max_depth': [10, 12, 15],
                  'min_samples_leaf': [20]}

clf = GridSearchCV(estimator=GBR,
                   param_grid=parameter_grid, cv=5)

clf.fit(X_train, y_train)

ypred = clf.best_estimator_.predict(X_val)
ypred = np.where(ypred<0, 0, ypred)
error_cal = pd.concat([y_val, pd.DataFrame(ypred)], axis = 1)
error_cal.columns = ['Actual', 'Predicted']
error = np.sqrt(np.mean((np.expm1(error_cal['Predicted'])/np.expm1(error_cal['Actual'])-1) ** 2))
print('RMSPE: {:.4f}'.format(error))
#0.1738

#------------Adding weight to the output of XGB to get best accuracy-----------

rmspe = {}
for w in np.arange(0.95, 1.05, 0.005):
    error = np.sqrt(np.mean(((np.expm1(error_cal['Predicted'])*w)/np.expm1(error_cal['Actual'])-1) ** 2))
    print('RMSPE: {:.4f}'.format(error))
    rmspe[w]=error

# 0.98 is the best weight

#------------------Predicting on the test dataset------------------------------
y_pred_final = np.expm1(model.predict(xgb.DMatrix(X_test)))
pred_final = y_pred_final*0.98
final = pd.concat([test['Id'],pd.DataFrame(pred_final)], axis=1)
final.columns = ['Id', 'Sales']
final.to_csv("final_results.csv", index=False)