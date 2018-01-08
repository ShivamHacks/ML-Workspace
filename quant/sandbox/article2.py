# https://www.datacamp.com/community/tutorials/finance-python-trading

import pandas as pd
import pandas_datareader as pdr
import datetime

aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(2017, 10, 1), end=datetime.datetime(2017, 11, 1))

# Get Data

def get(tickers, startdate, enddate):
  def data(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map (data, tickers)
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))

tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2017, 10, 1), datetime.datetime(2017, 11, 1))
print all_data