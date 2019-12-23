#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:22:33 2019

@author: verameowyh
"""

import tushare as ts
import pandas as pd
from pandas import Series 
import numpy
import time
import os
from datetime import datetime
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm


df = ts.get_hist_data('sh', start='2018-08-06',end='2019-12-15')

df = df.sort_values(by='date')

df = Series(df.close)

df2 = pd.read_csv('data/date_BI.csv', header=None)
df2.set_index(0, inplace=True)

from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):
    #rolling_statistics(timeseries)
    print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
ts_log = np.log(df)
ts_log.plot()

ts_log_diff = ts_log - ts_log.shift(periods=1)
ts_log_diff.plot()
plt.show()


#Y(k)=X(k+1)-X(k)
ts_log_diff = ts_log - ts_log.shift(periods=1)
ts_log_diff.dropna(inplace = True)
ts_log_diff2 = ts_log_diff - ts_log_diff.shift(periods=1)
ts_log_diff2.plot()
plt.show()

X = df.values
adfresult = adfuller(X)
print('ADF Statistic: %f' % adfresult[0])
print('p-value: %f' % adfresult[1])
print('Critical Values:')
for key, value in adfresult[4].items():
	print('\t%s: %.3f' % (key, value))


import statsmodels.api as sm
def acf_pacf_plot(ts_log_diff):
    sm.graphics.tsa.plot_acf(ts_log_diff,lags=40) #ARIMA,q
    sm.graphics.tsa.plot_pacf(ts_log_diff,lags=40) #ARIMA,p
acf_pacf_plot(ts_log_diff)


# Getting the optimal p and q
import sys
from statsmodels.tsa.arima_model import ARMA
def _proper_model(ts_log_diff, maxLag):
    best_p = 0 
    best_q = 0
    best_bic = sys.maxsize
    best_model=None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(ts_log_diff, order=(p, q))
            try:
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            print(bic, best_bic)
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    return best_p,best_q,best_model
_proper_model(ts_log_diff, 10)


merge = pd.merge(ts_log_diff,df2, how='inner', left_index=True, right_index=True)
ts_df = merge.iloc[:,0]
ts_df2 = merge.iloc[:,1]


arma_mod1 = sm.tsa.ARMA(ts_df, (1,0), ts_df2).fit()
predict_price1 = arma_mod1.predict(start=300, end=330, exog=ts_df2, dynamic=True)
print('Predicted Price (ARMAX): {}' .format(predict_price1))
plt.plot(ts_df["2019-11-04":])
plt.yticks([])
plt.xticks(["2019-11-04","2019-11-11","2019-11-18","2019-11-25","2019-12-02","2019-12-09"])
plt.title('Stock Index')
plt.plot(predict_price1)
plt.xticks(["2019-11-04","2019-11-11","2019-11-18","2019-11-25","2019-12-02","2019-12-09"])
plt.yticks([])
plt.title('Stock Index Forecasting')




