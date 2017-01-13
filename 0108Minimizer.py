"""Minimize an objective function, using Scipy."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

from util import get_data, plot_data


def f(X):
    Y = (X- 1.5)**2 +0.5
    print "X={},Y={}".format(X,Y)
    return Y

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    #daily_returns.ix[0,:] = 0 # set daily returns for row 0 to 0
    daily_returns.ix[0,0] = 0
    return daily_returns


def test_run():
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method='SLSQP',
        options={'disp': True})
    print "Minima found at:"
    print "X={}, Y={}".format(min_result.x, min_result.fun)

    # Plot function values, mark minima
    Xplot = np.linspace(0.5,2.5,21)
    Yplot = f(Xplot)
    plt.plot(Xplot,Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("Minima of an objective function")
    plt.show()



if __name__ == "__main__":
    test_run()