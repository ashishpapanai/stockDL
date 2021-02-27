import pandas as pd
from . import calculations, preprocessing, market


class Results:
    def __init__(self, ticker):
        self.preprocessing = preprocessing.data_preprocessing(ticker)
        self.market = market.Market(ticker)
        self.calculations = calculations.Calculations()
        self.result = self.result_calculations()

    def result_calculations(self):
        print("Test period of {:.2f} years, from {} to {} \n".format(len(self.preprocessing.v_bh)/12, 
                str(self.preprocessing.test.loc[self.preprocessing.test.index[0], "First Day Current Month"])[:10]
                , str(self.preprocessing.test.loc[self.preprocessing.test.index[-1], "First Day Next Month"])[:10]))

        results = pd.DataFrame({})
        results["Method"] = ["Buy and hold", "Moving average", "LSTM", "Mix"]
        vs = [self.preprocessing.v_bh, self.preprocessing.v_ma, self.market.v_lstm, self.market.v_mix]
        # print(vs)
        results["Total Gross Yield"] = [str(round(self.calculations.gross_yield(self.preprocessing.test, vi)[0], 2))+" %" for vi in vs]
        results["Annual Gross Yield"] = [str(round(self.calculations.gross_yield(self.preprocessing.test, vi)[1], 2))+" %" for vi in vs]
        results["Total Net Yield"] = [str(round(self.calculations.net_yield(self.preprocessing.test, vi)[0], 2))+" %" for vi in vs]
        results["Annual Net Yield"] = [str(round(self.calculations.net_yield(self.preprocessing.test, vi)[1], 2))+" %" for vi in vs]
        
        return results
        