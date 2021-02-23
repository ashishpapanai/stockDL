import numpy as np
import data

class data_preprocessing():
    def __init__(self):
        self.df_monthly = self.monthly_df(data.df)
    
    def monthly_df(df):

        dfm = df.resample("M").mean()
        dfm = dfm[:-1]

        dfm["First Day Current Month"] = data.first_days[:-1]
        dfm["First Day Next Month"] = data.first_days[1:]
        dfm["First Day Current Month Opening"] = np.array(
            df.loc[data.first_days[:-1], "Open"])
        dfm["First Day Next Month Opening"] = np.array(
            df.loc[data.first_days[1:], "Open"])
        dfm["Quotient"] = dfm["First Day Next Month Opening"].divide(
        dfm["First Day Current Month Opening"])

        dfm["mv_avg_12"] = dfm["Open"].rolling(window=12).mean().shift(1)
        dfm["mv_avg_24"] = dfm["Open"].rolling(window=24).mean().shift(1)

        dfm = dfm.iloc[24:, :]
        return dfm