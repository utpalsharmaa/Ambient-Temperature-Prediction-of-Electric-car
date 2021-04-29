# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:38:44 2021

@author: Admin
"""
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import matplotlib.pyplot as plt
from tkPDFViewer import tkPDFViewer as pdf
import matplotlib.pyplot as plt # plotting
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold,RFE,SelectKBest,f_regression,mutual_info_regression
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
from matplotlib import rcParams
from matplotlib.cm import rainbow
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, chi2
import sklearn.metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import  mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,BaggingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR, NuSVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.linear_model import Ridge, RidgeCV, SGDRegressor, ElasticNet, ElasticNetCV, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import keras
from keras.models import Sequential
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataset1 = pd.read_csv('motornew.csv')
df=dataset1
df.info()
prof_id = 6

curr_df = df[df['profile_id'] == prof_id]

curr_df = curr_df.drop('profile_id', axis = 1)
columns = curr_df.columns.tolist()
scaler = MinMaxScaler()

scurr_df = pd.DataFrame(scaler.fit_transform(curr_df), columns= columns)
scurr_df.head()

plt.title('Regression between motor speed and pm')
sns.regplot(x=curr_df['motor_speed'],y=curr_df['pm'], color='g')
plt.show()
plt.figure(figsize=(8, 8))
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),cmap='PiYG',annot=True)


#Before Feature Scaling
X = curr_df.drop(['pm'], axis = 1)
y = curr_df['pm']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set result
y_pred_logreg = regressor.predict(X_test)

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
EVC=explained_variance_score(y_test, y_pred_logreg)
ME=max_error(y_test, y_pred_logreg)
MAE=mean_absolute_error(y_test, y_pred_logreg)
MSE=mean_squared_error(y_test, y_pred_logreg)
MDAE=median_absolute_error(y_test, y_pred_logreg)
RS=r2_score(y_test, y_pred_logreg)
dfreg=pd.DataFrame(columns = ['Regressor','EVC','ME', 'MAE','MSE','MDAE','RS'])
#L2=['LinearRegressor','explained_variance_score','max_error', 'mean_absolute_error','mean_squared_error','median_absolute_error','r2_score']
#dfreg.loc[len(dfreg),:]=L2
L2=["LinearRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2



from sklearn.ensemble import AdaBoostRegressor
adaregressor = AdaBoostRegressor()
adaregressor.fit(X_train, y_train)

# predicting the test set result
y_pred_adareg = adaregressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_adareg)
ME=max_error(y_test, y_pred_adareg)
MAE=mean_absolute_error(y_test, y_pred_adareg)
MSE=mean_squared_error(y_test, y_pred_adareg)
MDAE=median_absolute_error(y_test, y_pred_adareg)
RS=r2_score(y_test, y_pred_adareg)
L2=["AdaBoostRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2


from sklearn.ensemble import BaggingRegressor
baregressor = BaggingRegressor()
baregressor.fit(X_train, y_train)

# predicting the test set result
y_pred_bareg = baregressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_bareg)
ME=max_error(y_test, y_pred_bareg)
MAE=mean_absolute_error(y_test, y_pred_bareg)
MSE=mean_squared_error(y_test, y_pred_bareg)
MDAE=median_absolute_error(y_test, y_pred_bareg)
RS=r2_score(y_test, y_pred_bareg)
L2=["BaggingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2




from sklearn.ensemble import ExtraTreesRegressor
etregressor = ExtraTreesRegressor()
etregressor.fit(X_train, y_train)

# predicting the test set result
y_pred_etreg = etregressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_etreg)
ME=max_error(y_test, y_pred_etreg)
MAE=mean_absolute_error(y_test, y_pred_etreg)
MSE=mean_squared_error(y_test, y_pred_etreg)
MDAE=median_absolute_error(y_test, y_pred_etreg)
RS=r2_score(y_test, y_pred_etreg)
L2=["ExtraTreesRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2



from sklearn.ensemble import GradientBoostingRegressor
gbregressor = GradientBoostingRegressor()
gbregressor.fit(X_train, y_train)

# predicting the test set result
y_pred_gbreg = gbregressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_gbreg)
ME=max_error(y_test, y_pred_gbreg)
MAE=mean_absolute_error(y_test, y_pred_gbreg)
MSE=mean_squared_error(y_test, y_pred_gbreg)
MDAE=median_absolute_error(y_test, y_pred_gbreg)
RS=r2_score(y_test, y_pred_gbreg)
L2=["GradientBoostingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2



from sklearn.ensemble import RandomForestRegressor
rfregressor = RandomForestRegressor()
rfregressor.fit(X_train, y_train)

# predicting the test set result
y_pred_rfreg = rfregressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_rfreg)
ME=max_error(y_test, y_pred_rfreg)
MAE=mean_absolute_error(y_test, y_pred_rfreg)
MSE=mean_squared_error(y_test, y_pred_rfreg)
MDAE=median_absolute_error(y_test, y_pred_rfreg)
RS=r2_score(y_test, y_pred_rfreg)
L2=["RandomForestRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2



from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
regr = ElasticNet(random_state=0)
regr.fit(X_train, y_train)
y_pred_enreg = regr.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_enreg)
ME=max_error(y_test, y_pred_enreg)
MAE=mean_absolute_error(y_test, y_pred_enreg)
MSE=mean_squared_error(y_test, y_pred_enreg)
MDAE=median_absolute_error(y_test, y_pred_enreg)
RS=r2_score(y_test, y_pred_enreg)
L2=["ElasticNet",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2



from sklearn.kernel_ridge import KernelRidge
krregressor = KernelRidge()
krregressor.fit(X_train, y_train)

# predicting the test set result
y_pred_krreg =krregressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_krreg)
ME=max_error(y_test, y_pred_krreg)
MAE=mean_absolute_error(y_test, y_pred_krreg)
MSE=mean_squared_error(y_test, y_pred_krreg)
MDAE=median_absolute_error(y_test, y_pred_krreg)
RS=r2_score(y_test, y_pred_krreg)
L2=["KernelRidge",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2




from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf', gamma='auto')
svr_regressor.fit(X_train, y_train)
y_pred_svreg =svr_regressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_svreg)
ME=max_error(y_test, y_pred_svreg)
MAE=mean_absolute_error(y_test, y_pred_svreg)
MSE=mean_squared_error(y_test, y_pred_svreg)
MDAE=median_absolute_error(y_test, y_pred_svreg)
RS=r2_score(y_test, y_pred_svreg)
L2=["SVR",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2


from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(X_train, y_train)
y_pred_dtreg =tree_regressor.predict(X_test)

EVC=explained_variance_score(y_test, y_pred_dtreg)
ME=max_error(y_test, y_pred_dtreg)
MAE=mean_absolute_error(y_test, y_pred_dtreg)
MSE=mean_squared_error(y_test, y_pred_dtreg)
MDAE=median_absolute_error(y_test, y_pred_dtreg)
RS=r2_score(y_test, y_pred_dtreg)
L2=["DecisionTreeRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfreg.loc[len(dfreg),:]=L2






# WITH FEATURE SCALING

X = curr_df.drop(['pm'], axis = 1)
y = curr_df['pm']

sc_X = StandardScaler()
X= sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_logreg = regressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_logreg)
ME=max_error(y_test, y_pred_logreg)
MAE=mean_absolute_error(y_test, y_pred_logreg)
MSE=mean_squared_error(y_test, y_pred_logreg)
MDAE=median_absolute_error(y_test, y_pred_logreg)
RS=r2_score(y_test, y_pred_logreg)
dfregfs=pd.DataFrame(columns = ['Regressor','EVC','ME', 'MAE','MSE','MDAE','RS'])
#L2=['LinearRegressor','explained_variance_score','max_error', 'mean_absolute_error','mean_squared_error','median_absolute_error','r2_score']
#dfreg.loc[len(dfreg),:]=L2
L2=["LinearRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2



from sklearn.ensemble import AdaBoostRegressor
adaregressor = AdaBoostRegressor()
adaregressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_adareg = adaregressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_adareg)
ME=max_error(y_test, y_pred_adareg)
MAE=mean_absolute_error(y_test, y_pred_adareg)
MSE=mean_squared_error(y_test, y_pred_adareg)
MDAE=median_absolute_error(y_test, y_pred_adareg)
RS=r2_score(y_test, y_pred_adareg)
L2=["AdaBoostRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2




from sklearn.ensemble import BaggingRegressor
baregressor = BaggingRegressor()
baregressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_bareg = baregressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_bareg)
ME=max_error(y_test, y_pred_bareg)
MAE=mean_absolute_error(y_test, y_pred_bareg)
MSE=mean_squared_error(y_test, y_pred_bareg)
MDAE=median_absolute_error(y_test, y_pred_bareg)
RS=r2_score(y_test, y_pred_bareg)
L2=["BaggingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2



from sklearn.ensemble import ExtraTreesRegressor
etregressor = ExtraTreesRegressor()
etregressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_etreg = etregressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_etreg)
ME=max_error(y_test, y_pred_etreg)
MAE=mean_absolute_error(y_test, y_pred_etreg)
MSE=mean_squared_error(y_test, y_pred_etreg)
MDAE=median_absolute_error(y_test, y_pred_etreg)
RS=r2_score(y_test, y_pred_etreg)
L2=["ExtraTreesRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2



from sklearn.ensemble import GradientBoostingRegressor
gbregressor = GradientBoostingRegressor()
gbregressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_gbreg = gbregressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_gbreg)
ME=max_error(y_test, y_pred_gbreg)
MAE=mean_absolute_error(y_test, y_pred_gbreg)
MSE=mean_squared_error(y_test, y_pred_gbreg)
MDAE=median_absolute_error(y_test, y_pred_gbreg)
RS=r2_score(y_test, y_pred_gbreg)
L2=["GradientBoostingRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2



from sklearn.ensemble import RandomForestRegressor
rfregressor = RandomForestRegressor()
rfregressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_rfreg = rfregressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_rfreg)
ME=max_error(y_test, y_pred_rfreg)
MAE=mean_absolute_error(y_test, y_pred_rfreg)
MSE=mean_squared_error(y_test, y_pred_rfreg)
MDAE=median_absolute_error(y_test, y_pred_rfreg)
RS=r2_score(y_test, y_pred_rfreg)
L2=["RandomForestRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2




from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
regr = ElasticNet(random_state=0)
regr.fit(SX_train, y_train)
y_pred_enreg = regr.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_enreg)
ME=max_error(y_test, y_pred_enreg)
MAE=mean_absolute_error(y_test, y_pred_enreg)
MSE=mean_squared_error(y_test, y_pred_enreg)
MDAE=median_absolute_error(y_test, y_pred_enreg)
RS=r2_score(y_test, y_pred_enreg)
L2=["ElasticNet",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2



from sklearn.kernel_ridge import KernelRidge
krregressor = KernelRidge()
krregressor.fit(SX_train, y_train)

# predicting the test set result
y_pred_krreg =krregressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_krreg)
ME=max_error(y_test, y_pred_krreg)
MAE=mean_absolute_error(y_test, y_pred_krreg)
MSE=mean_squared_error(y_test, y_pred_krreg)
MDAE=median_absolute_error(y_test, y_pred_krreg)
RS=r2_score(y_test, y_pred_krreg)
L2=["KernelRidge",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2




from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf', gamma='auto')
svr_regressor.fit(SX_train, y_train)
y_pred_svreg =svr_regressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_svreg)
ME=max_error(y_test, y_pred_svreg)
MAE=mean_absolute_error(y_test, y_pred_svreg)
MSE=mean_squared_error(y_test, y_pred_svreg)
MDAE=median_absolute_error(y_test, y_pred_svreg)
RS=r2_score(y_test, y_pred_svreg)
L2=["SVR",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2



from sklearn.tree import DecisionTreeRegressor
tree_regressor = DecisionTreeRegressor(random_state = 0)
tree_regressor.fit(SX_train, y_train)
y_pred_dtreg =tree_regressor.predict(SX_test)

EVC=explained_variance_score(y_test, y_pred_dtreg)
ME=max_error(y_test, y_pred_dtreg)
MAE=mean_absolute_error(y_test, y_pred_dtreg)
MSE=mean_squared_error(y_test, y_pred_dtreg)
MDAE=median_absolute_error(y_test, y_pred_dtreg)
RS=r2_score(y_test, y_pred_dtreg)
L2=["DecisionTreeRegressor",EVC,ME,MAE,MSE,MDAE,RS]
dfregfs.loc[len(dfregfs),:]=L2








