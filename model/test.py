from matplotlib.pyplot import plot
import data, preprocessing,calculations, models, train, plots, results, market


ticker = 'GOOGL'

data_preprocessor= preprocessing.data_preprocessing(ticker)
df_monthly = data_preprocessor.df_monthly
#data_loader = data.Data_Loader(ticker)
print(data_preprocessor.data_reader.end_date)
df_monthly = data_preprocessor.df_monthly
print(df_monthly.head())

X, y = data_preprocessor.data_scaling(df_monthly)

#print(X.shape, y.shape)
#training = train.Training(ticker)

train = train.Training(ticker)
train.train_model()

plots = plots.Plots(ticker)
plots.comparison_plots()
results = results.Results(ticker)

print(results.result_calculations())