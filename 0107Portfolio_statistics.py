"""Portfolio Statistics."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from util import get_data, plot_data

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    #daily_returns.ix[0,:] = 0 # set daily returns for row 0 to 0
    daily_returns.ix[0,0] = 0
    return daily_returns


def test_run():
    # Read data
    dates = pd.date_range('2012-05-18', '2016-11-30')
    symbols = ['GOOGL','FB','AMZN','IBM']
    df = get_data(symbols, dates)
    start_value = 1000000
    allocs = [0.4,0.4,0.1,0.1]

    # Plot the stock prices at first
    #plot_data(df, title="stock", ylabel="stock price")

    # normalize data frame with first row
    df_normed = df / df.ix[0,:]
    #df_normed = df / df.ix[0]
    
    # Allocate the allocation value to data frame
    df_alloced = df_normed * allocs

    # Position values
    df_pos_values = df_alloced * start_value

    # Portfolio Values
    df_port_values = df_pos_values.sum(axis = 1)

    #print df_port_values.tail

    # Plot the portfolio
    #plot_data(df_port_values,title="Portfolio Trends", ylabel="Portfolio")


    # compute the daily returns
    df_port_daily_returns = compute_daily_returns(df_port_values)
    plot_data(df_port_daily_returns,title="Portfolio Daily Returns", ylabel="Portfolio")
    # excludes 0 value 1st row
    df_port_daily_returns = df_port_daily_returns[1:]

    cumulative_returns = (df_port_values[-1]/df_port_values[0]) -1
    average_daily_returns = df_port_daily_returns.mean()
    std_daily_returns = df_port_daily_returns.std()

    print "Cumulative Return={}, Average Daily Return={}, Risk={}".format(
        cumulative_returns,average_daily_returns,std_daily_returns)

    # Sharp Ratios
    sharp_ratio = average_daily_returns / std_daily_returns
    print "Sharp Ratio = {}".format(sharp_ratio)



if __name__ == "__main__":
    test_run()