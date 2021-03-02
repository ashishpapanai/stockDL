'''
This module comprises all functions to calculate the net yield and gross yield.
'''
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import pandas as pd
import numpy as np

class Calculations():
    def __init__(self):
        self.capital_gains_tax = 0.10
        self.broker_comission = 0.003

    ''' This function will calculate the percentage gross yield in the investment per share of a particular stock ticker. '''
    def gross_yield(self, df, v):
        prod = (v*df["Quotient"]+1-v).prod()
        n_years = len(v)/12
        return (prod-1)*100, ((prod**(1/n_years))-1)*100

    '''
    Function to convert a 1D vector of zeros and ones to a 2D vector of zeros and ones,
    with the groups of ones seperated to different columns.
    E.g. [0,1,1,0,1,1,1,0,1]
    [[0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]]),

    This function will help us to separate In - Trading Months from the Out - Non Trading months.

    '''
    def separate_ones(self, u):
        u_ = np.r_[0, u, 0]
        i = np.flatnonzero(u_[:-1] != u_[1:])
        v, w = i[::2], i[1::2]
        if len(v) == 0:
            return np.zeros(len(u)), 0

        n, m = len(v), len(u)
        o = np.zeros(n*m, dtype=int)

        r = np.arange(n)*m
        o[v+r] = 1

        if w[-1] == m:
            o[w[:-1]+r[:-1]] = -1
        else:
            o[w+r] -= 1

        out = o.cumsum().reshape(n, -1)
        return out, n

    '''
    The following function will calculate the net yield in the share till the investment period.
    After deducting the capital gains and broker commission from the gains. 
    '''
    def net_yield(self, df, v):
        n_years = len(v)/12
        w, n = self.separate_ones(v)
        #print((w*np.array(df["Quotient"])+(1-w)).shape)
        A = (w*np.array(df["Quotient"])+(1-w)).prod()
        A1p = np.maximum(0, np.sign(A-1))
        Ap = A*A1p
        Am = A-Ap
        An = Am+(Ap-A1p)*(1-self.capital_gains_tax)+A1p
        prod = An.prod()*((1-self.broker_comission)**(2*n))
        return (prod-1)*100, ((prod**(1/n_years))-1)*100

    '''
    Total money including the basic invested by the investor in a share in the market.     
    '''
    def gross_portfolio(self, df, w):
        portfolio = [(w*df["Quotient"]+(1-w))[:i].prod() for i in range(len(w))]
        return portfolio
