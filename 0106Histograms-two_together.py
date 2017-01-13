"""Plot histograms in one chart."""

import pandas as pd
import matplotlib.pyplot as plt

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

    # Plot histograms on the same chart
    daily_returns['GOOGL'].hist(bins=20, label = "GOOGL")
    daily_returns['FB'].hist(bins=20, label = "FB")
    daily_returns['AMZN'].hist(bins=20, label = "AMZN")
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":
    test_run()