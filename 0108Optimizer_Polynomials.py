"""Fit a line to a given set of data points using optimization."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

from util import get_data, plot_data


def error(C, data): #error function
    """Compute error between given polynomial model and observed data

    Parameters
    -----------
    C: numpy.polyld object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x,y)

    Returns error as a single real value.
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:,1] - np.polyval(C, data[:,0]))**2)
    return err

def fit_poly(data, error_func, degree=3):
    """ Fit a polynomial to given data, using a supplied error function

    Parameters
    ---------------
    data: 2D array where each row is a point (X,Y)
    error_func: function that computes the error between a polynomial and obeserved data

    Returns polynomial that minimizes the error function.
    """
    # Generate initial guess for polynomial model (all coeffs =1)
    Cguess=np.poly1d(np.ones(degree + 1, dtype=np.float32)) 

    # Plot initial guess (optional)
    x = np.linspace(-5,5,21)
    plt.plot(x, np.polyval(Cguess,x),'m--', linewidth=2.0, 
        label="Initial guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, Cguess, args=(data), method='SLSQP',
        options={'disp': True})
    return np.poly1d(result.x)


def test_run():
    # Define original polynomial
    p_orig = np.float32([1.5,-10,-5,60,50])
    print "Original polynomial:"
    Xorig = np.linspace(0,30,41)
    Yorig = p_orig[0] * (Xorig**4)+ p_orig[1] * (Xorig**3)+ p_orig[2] * Xorig**2 +p_orig[3] * Xorig+p_orig[4]

    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original Polynomial")

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label="Data Points")

    # Try to fit a line to this data
    p_fit = fit_poly(data,error)
    print "Fitted Polynomial: C0={}, C1={},C2={},C3={},C5={}".format(p_fit[0],p_fit[1],p_fit[2],p_fit[3],p_fit[4])
    plt.plot(data[:,0], p_fit[0]*data[:,0]**4 + p_fit[1]*data[:,0]**3+p_fit[2]*data[:,0]**2+p_fit[3]*data[:,0] + p_fit[4],
        'r--',linewidth=2.0, label="Fit Polynomial")

    plt.title("Best fit Polynomial")
    plt.legend(loc='upper left')
    plt.show()



if __name__ == "__main__":
    test_run()