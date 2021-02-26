import pandas as pd
import numpy as np

class Calculations():
    def __init__(self):
        self.capital_gains_tax = 0.10
        self.broker_comission = 0.003

    def gross_yield(self, df, v):
        prod = (v*df["Quotient"]+1-v).prod()
        n_years = len(v)/12
        return (prod-1)*100, ((prod**(1/n_years))-1)*100

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

    def gross_portfolio(self, df, w):
        portfolio = [(w*df["Quotient"]+(1-w))[:i].prod() for i in range(len(w))]
        return portfolio
