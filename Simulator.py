from statistics import stdev
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#take input
ticker = (input('Input ticker: ')).upper()
data_source = (input('Input data Source: ')).lower()
start = input('Input Start date (YYYY-MM-DD) (No leading Zeros): ')
days = input('Days to calculate: ')

#Get data
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source, start)['Adj Close']

data.plot(figsize = (15,6))
plt.show()

#for calculating return
log_returns = np.log(1 + data.pct_change())

#Get data to graph
sns.distplot(log_returns.iloc[1:])
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.show()

u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5*var)

stdev = log_returns.std()
days = int(days)
trials = 10000
#ppf (inverse cdf), given some percent return an x value
z = norm.ppf(np.random.rand(days,trials))
#measures dollar change of stock price with random variables
daily_returns = np.exp(drift.values + stdev.values * z)

#calculate stock price
price_paths = np.zeros_like(daily_returns)
price_paths[0]=data.iloc[-1]
for t in range (1,days):
    #price today = Price Yesterday * e^r
    price_paths[t] = price_paths[t-1]*daily_returns[t]

#Generate graph of Monte Carlo simulation
plt.figure(figsize = (15,6))
plt.plot(pd.DataFrame(price_paths).iloc[:,0:days])
plt.show()

#Generate Histogram of last-day prices
sns.distplot(pd.DataFrame(price_paths).iloc[-1])
plt.xlabel("Price after " + str(days) + " days")
plt.show()