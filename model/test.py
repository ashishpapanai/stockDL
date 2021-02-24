import data, preprocessing,calculations, models, train


ticker = 'MSFT'
data_preprocessor= preprocessing.data_preprocessing(ticker)
df_monthly = data_preprocessor.df_monthly
#data_loader = data.Data_Loader(ticker)
print(data_preprocessor.data_reader.end_date)
df_monthly = data_preprocessor.df_monthly
print(df_monthly.head())

X, y = data_preprocessor.data_scaling(df_monthly)
print(X.shape, y.shape)

training = train.Training(ticker)