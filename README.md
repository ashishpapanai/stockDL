![Stock (1)](https://user-images.githubusercontent.com/52123364/109387767-49942e00-7929-11eb-92d6-970f5bc81107.png)
# stockDL: A Deep Learning library for stocks price predictions and calculations
[![PyPI version](https://badge.fury.io/py/stockDL.svg)](https://pypi.org/project/stockDL/)    [![Documentation Status](https://readthedocs.org/projects/stockdl/badge/?version=latest)](https://stockdl.readthedocs.io/en/latest/?badge=latest) ![](https://img.shields.io/github/stars/ashishpapanai/stockDL.svg) ![](https://img.shields.io/github/forks/ashishpapanai/stockDL.svg) ![](https://img.shields.io/github/tag/ashishpapanai/stockDL.svg) ![](https://img.shields.io/github/release/ashishpapanai/stockDL.svg) ![](https://img.shields.io/github/issues/ashishpapanai/stockDL.svg) 

> Copy paste is not the way you should share code.

### Features

- **Single stock trading and price comparisons** based on 2 traditional stock market algorithms [Buy and Hold & Moving Average], and 2 deep learning algorithms [LSTM Network and Conv1D + LSTM Network]
- **Returns result in JSON** format comprising the Total Gross Yield, Annual Gross Yield, Total Net Yield, and Annual Net Yield. This JSON Result can be used for web-based price predictions. Considering the broker commission and capital gains tax in India *[can be modified]*
- **Dynamic model training** every time the library is run thus making the model unaffected by unusual stock market changes due to Act of God, Pandemics, Sudden loss, Gains to the share prices.
- **Latest Financial Data** collection from Yahoo Finance API (from the starting date of the stock to the current data).
- **Easy backend integration** with flask or another python backend for web deployment. 
- **Less than 90 seconds result processing time** on Tesla K80 GPU with 4992 NVIDIA CUDA and 24 GB VRAM. Much faster than other deep learning stocks analysers.  *[can be used on Google Colab]*
- **Easy Installation** with pip. Install and run. Dependencies satisfied automatically.
- **Different plots available** according to the user requirements, Plots to show the training and validation accuracy, Months to trade in the market and months to stay out, comparison of the 4 trading strategies and the market predictions for the coming month.

### How to install:
##### For using as a library:
`pip install stockDL`

import the package as:
```py 
import stockDL
```

to get the results in command line:
```py
from stockDL import main
main.Main('stock_ticker')
```
Stock tickers can be obtained [here.](http://https://finance.yahoo.com/ "here.")

##### For using as a template or to make contributions to the repository:
Clone from GitHub: https://github.com/ashishpapanai/stockDL

```sh 
git clone https://github.com/ashishpapanai/stockDL
```
Create a virtual environment using pip for Linux and macOS:
```sh
python3 -m pip install --user virtualenv
# Create a virtual environment
python3 -m venv env
# Activate the virtual environment
source env/bin/activate
```
Create a virtual environment using pip for Windows:
```sh
py -m pip install --user virtualenv
# Create a virtual environment
py -m venv env
# Activate the virtual environment
.\env\Scripts\activate
```
Installing dependencies:
``` 
pip install -r requirements.txt
```
Running the package: 
```
python -m stockDL
```
### Dependencies:
1. Yahoo Finance (yfinance): https://pypi.org/project/yfinance/
2. Keras: https://pypi.org/project/Keras/
3. Pandas: https://pypi.org/project/pandas/
4. Numpy: https://pypi.org/project/numpy/
5. Matplotlib: https://pypi.org/project/matplotlib/
6. TensorFlow: https://pypi.org/project/tensorflow/

Install all dependencies in a go: `pip install -r requirements.txt`
If it fails: Install all dependencies on by one [if you are cloning the repository]. 
for pip installation, dependencies are satisfied automatically. 

### License:
[MIT License &copy; Ashish Papanai 2021](https://github.com/ashishpapanai/stockDL/blob/master/LICENSE "MIT License Ashish Papanai 2021")

### Documentation: 
Coming Soon! 

### Getting Help: 
Post your questions in the [discussion section](https://github.com/ashishpapanai/stockDL/discussions "discussion section") of the GitHub repository or mail the author [ashishpapanai@gmail.com]

### Contributing to stockDL:
Contributions are not restricted to bug fixes or enhancements. We welcome contributions including any grammatical or typo error anywhere in the repository. 

You can contribute by reviewing the PRs, requesting new and useful features, reporting a bug in the repository or helping the community in the discussion section. 


###### Copyright &copy; Ashish Papanai 2021
