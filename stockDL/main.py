from . import preprocessing, train, plots, results


class Main:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_preprocessor = preprocessing.data_preprocessing(ticker)
        self.df_monthly = self.data_preprocessor.df_monthly
        print(self.data_preprocessor.data_reader.end_date)
        df_monthly = self.data_preprocessor.df_monthly
        # X, y = self.data_preprocessor.data_scaling(df_monthly)
        self.train = train.Training(ticker)
        self.train.train_model()
        self.plots = plots.Plots(ticker)
        self.plots.comparison_plots()
        self.results = results.Results(ticker)
        self.result = self.results.result
        self.result_json = self.result.to_json(orient="split")
        print(self.result)
        print(self.result_json)
