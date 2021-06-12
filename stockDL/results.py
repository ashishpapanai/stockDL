'''
This is the final module responsible for the result calculation and parsing the dataframe to JSON.
This module requires data from preprocessing, calculations, market and all their dependencies.
'''
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import pandas as pd
from . import calculations, preprocessing, market

class Results:
    def __init__(self, ticker):
        self.preprocessing = preprocessing.data_preprocessing(ticker)
        self.market = market.Market(ticker)
        self.calculations = calculations.Calculations()
        self.result = self.result_calculations()

    '''
    This function handles the result calculation based on the formula devised in the calculation module,
    It generates the total yield in a 6-year investment and annual yields as well.
    '''
    def result_calculations(self):
        print("Test period of {:.2f} years, from {} to {} \n".format(len(self.preprocessing.v_bh) / 12,
                                                                     str(self.preprocessing.test.loc[
                                                                             self.preprocessing.test.index[0],
                                                                             "First Day Current Month"])[:10],
                                                                     str(self.preprocessing.test.loc[
                                                                             self.preprocessing.test.index[-1],
                                                                             "First Day Next Month"])[:10]))

        results = pd.DataFrame({})
        results["Method"] = ["Buy and hold", "Moving average", "LSTM", "Mix"]
        ''' This list stores the market data predictions obtained from the market module. '''
        vs = [self.preprocessing.v_bh, self.preprocessing.v_ma, self.market.v_lstm, self.market.v_mix]
        results["Total Gross Yield"] = [
            str(round(self.calculations.gross_yield(self.preprocessing.test, vi)[0], 2)) + " %" for vi in vs]
        results["Annual Gross Yield"] = [
            str(round(self.calculations.gross_yield(self.preprocessing.test, vi)[1], 2)) + " %" for vi in vs]
        results["Total Net Yield"] = [str(round(self.calculations.net_yield(self.preprocessing.test, vi)[0], 2)) + " %"
                                      for vi in vs]
        results["Annual Net Yield"] = [str(round(self.calculations.net_yield(self.preprocessing.test, vi)[1], 2)) + " %"
                                       for vi in vs]

        #results_json = results.to_json(orient="records")
        #parsed = json.loads(result)
        #json.dumps(parsed, indent=4) 
        #print(results_json)
        return results
