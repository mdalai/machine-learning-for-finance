"""Scatterplots."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from util import get_data, plot_data

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.ix[0,:] = 0 # set daily returns for row 0 to 0
    return daily_returns


def test_run():
    # Read data
    dates = pd.date_range('2011-01-01', '2016-12-31')
    symbols = ['GOOGL','FB','AMZN']
    df = get_data(symbols, dates)
    #plot_data(df,title="Stock Price", ylabel="Price")

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    #plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")

    # Scatterplot 
    daily_returns.plot(kind='scatter',x='GOOGL', y="AMZN")
    beta_FB, alpha_FB = np.polyfit(daily_returns['GOOGL'],daily_returns['AMZN'], 1)
    print beta_FB, alpha_FB
    plt.plot(daily_returns['GOOGL'],beta_FB*daily_returns['GOOGL']+alpha_FB,'-',color='r')
    #plt.plot(daily_returns['GOOGL'],beta_FB*daily_returns['GOOGL']+alpha_FB,'.r-')
    plt.show()

    # Calculate correlation coefficient
    print daily_returns.corr(method='pearson')




if __name__ == "__main__":
    test_run()