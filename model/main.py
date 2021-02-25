import model.preprocessing, model.train, model.plots, model.results
print("done")
class finDL():
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_preprocessor = model.preprocessing.data_preprocessing(ticker)
        self.df_monthly = self.data_preprocessor.df_monthly
        print(self.data_preprocessor.data_reader.end_date)
        df_monthly = self.data_preprocessor.df_monthly
        #print(df_monthly.head())
        X, y = self.data_preprocessor.data_scaling(df_monthly)
        self.train = model.train.Training(ticker)
        self.train.train_model()
        self.plots = model.plots.Plots(ticker)
        self.plots.comparison_plots()
        self.results = model.results.Results(ticker)
        self.result = self.results.result
        self.result_json = self.result.to_json(orient="split")
        print(self.result)
        print(self.result_json)