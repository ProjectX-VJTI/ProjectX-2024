# Stock Transformers

A Project by [Khush Agrawal](https://github.com/Khushmagrawal), [Yashvi Mehta](https://github.com/YashviMehta03) and [Ghruank Kothare](https://github.com/Ghruank) and mentored by [Tvisha Vedant](https://github.com/tvilight4) and [Kindipsingh Malhi](https://github.com/kindipsingh)

[Project Blog](https://yashvimehta03.github.io/stock-transformers-blog/)

## 📌 Aim

In the modern capital market, the price of a stock is often considered to be highly volatile and unpredictable because of various social, financial, political and other dynamic factors that can bring catastrophic financial loss to the investors. This project aims to predict stock prices using transformer architecture by utilising the concept of time series forecasting.

The transformer model has been widely leveraged for natural language processing and computer vision tasks,but is less frequently used for tasks like stock prices prediction. The introduction of time2vec encoding to represent the time series features has made it possible to employ the transformer model for the stock price prediction. We aim to leverage these two effective techniques to discover forecasting ability on volatile stock markets.

## 🗺️ Overview

In this project :

1) We have classified stocks into 2 categories, volatile and non-volatile.
2) For each of them we worked on 3 models namely , LSTM, transformer and informer.
3) We perfomed feature engineering and time2vec to create our datasets.
4) Our models predict the closing price of a stock on a day based on data of previous 100 days.
5) Finally, we concluded it by comparing results across all models and determining which one had the best accuracy.

## ⚙️ Tech Stack

| Category                      | Technologies                                                                                                                                                                                                                  |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Programming Languages**      | ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)                                                                                                                                   |
| **Frameworks**                 | ![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)                                                                                                                        |
| **Libraries**                  | ![YFinance](https://img.shields.io/badge/-YFinance-013220?style=flat) ![Keras](https://img.shields.io/badge/-Keras-D00000?style=flat&logo=keras&logoColor=white)                                                               |
| **Deep Learning Models**       | ![LSTM](https://img.shields.io/badge/-LSTM-FF9900?style=flat&logo=tensorflow&logoColor=white) ![Transformers](https://img.shields.io/badge/-Transformers-FF9900?style=flat&logo=tensorflow&logoColor=white) ![Informer](https://img.shields.io/badge/-Informer-343434?style=flat)|
| **Tools**                      | ![Git](https://img.shields.io/badge/-Git-F05032?style=flat&logo=git&logoColor=white) ![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) ![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white) |
| **Visualization & Analysis**   | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=python&logoColor=white)                                                                                                                           |

## 📂 File Structure

### Root
```
├── README.md 
│ 
├── Assignments 
│   ├── C2W1_Gradient_Checking.ipynb  
│   ├── C2W1_Initialization.ipynb  
│   ├── C2W1_Regularization.ipynb 
│   ├── C2W2_Optimization_methods.ipynb
│   ├── C2W3_Tensorflow_introduction.ipynb
│   ├── Logistic_Regression_with_a_Neural_Network_mindset.ipynb
│   ├── Planar_data_classification_with_one_hidden_layer.ipynb
│   ├── Python_Basics_with_Numpy.ipynb
│   └── README.md
│
├── informer_shuffle
│   └── informer_code
│       └── Informer2020-main
│           ├── Dockerfile
│           ├── environment.yml
│           ├── LICENSE
│           ├── main_informer.ipynb
│           ├── main_informer.py
│           ├── Makefile
│           ├── README.md
│           ├── requirements.txt
│           │
│           ├── data
│           │   ├── data_loader.py
│           │   ├── infosys (1).csv
│           │   └── __init__.py
│           │
│           ├── exp
│           │   ├── exp_basic.py
│           │   ├── exp_informer.py
│           │   └── __init__.py
│           │
│           ├── img
│           │   ├── data.png
│           │   ├── informer.png
│           │   ├── probsparse_intro.png
│           │   ├── result_multivariate.png
│           │   └── result_univariate.png
│           │
│           ├── models
│           │   ├── attn.py
│           │   ├── decoder.py
│           │   ├── embed.py
│           │   ├── encoder.py
│           │   ├── model.py
│           │   └── __init__.py
│           │
│           ├── scripts
│           │   ├── ETTh1.sh
│           │   ├── ETTh2.sh
│           │   ├── ETTm1.sh
│           │   └── WTH.sh
│           │
│           └── utils
│               ├── masking.py
│               ├── metrics.py
│               ├── timefeatures.py
│               ├── tools.py
│               └── __init__.py
│
├── LSTM
│   └── lstm_t2v.ipynb
│
├── mini-projects
│   ├── miniproj_final.ipynb
│   └── monthly_milk_production_1.csv
│
├── Notes
│   ├── Notes_01-08-24.md
│   ├── Notes_18-07-24.md
│   ├── Notes_19-07-24.md
│   ├── Notes_21-07-24.md
│   ├── Notes_23-07-24.md
│   ├── Notes_24-07-24.md
│   ├── Notes_25-07-24.md
│   ├── Notes_28-07-24.md
│   └── Notes_31_07_24.md
│   │
│   └── images
│
└── Transformer
    ├── .txt
    └── transformer.ipynb
```

## 🔨 Data Extracting 

We used the yFinance Python Library for accessing financial data​.  

Features Extracted -> Open, high, low, close, volume, adj close

## ⌛ Time Series Forecasting

Time series forecasting is a statistical or machine learning technique that uses historical and current data to predict future values over a period of time or a specific point in the future. It involves building models from historical data and using them to make observations

Forecasting has a range of applications in various industries especially the stock market!

## ⌚ Time2Vec

Time2Vec is a time encoding mechanism that transforms time-related features into a higher-dimensional space, capturing both linear and periodic patterns. It uses a combination of sine and linear components to effectively represent temporal information. This encoding helps improve the performance of models in time series forecasting tasks by providing a richer representation of time.

## 💾 LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can process and analyze sequential data, such as text, speech, and time series. LSTMs are well-suited for problems involving large data sequences and can identify both short-term and long-term dependencies.

### Why LSTM for stock predictions ?
1) Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that can process and analyze sequential data, such as time series. ​

2) Stock prices, are non-stationary and exhibit trends and seasonality. LSTMs can handle these non-linear relationships within the data.​

3) LSTM's can learn long and short term dependencies by selectively retaining information through the memory cell and gates.

4) This characteristic is particularly beneficial in financial time series analysis, where understanding previous market trends is vital for forecasting future price changes. By maintaining a memory cell that stores relevant information over extended periods, LSTMs demonstrate superior performance in capturing subtle nuances and trends within complex trading datasets.

5) Our LSTM model takes 5 features namely, Open, High, Low, Adjusted close and Volume and label is the Closing Price.

## 📃 Feature Engineering

Feature engineering is the process of selecting, manipulating and transforming raw data into features that can be used in supervised learning.

It requires using domain knowledge to select and transform the most relevant variables

The goal of feature engineering and selection is to improve the performance of machine learning (ML) algorithms.

### Features Added
1) RSI - Relative Strength Index
The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price changes to detect overvalued or undervalued conditions in the price of that security.

On a scale of 0 to 100, RSI reading of 70 or above indicates an overbought condition. A reading of 30 or below indicates an oversold condition.

2) ROC - rate of change
The price rate of change (ROC) is a momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago.

The ROC indicator is plotted against zero, with the indicator moving upwards into positive territory if price changes are to the upside, and moving into negative territory if price changes are to the downside.

It can be used to spot divergences, overbought and oversold conditions, and centerline crossovers.

3) Bollinger bands
Bollinger Bands help gauge the volatility of stocks to determine if they are over or undervalued.

Bands appear on stock charts as three lines that move with the price. The center line is the stock price's 20-day simple moving average (SMA). The upper and lower bands are set at a certain number of standard deviations, usually two, above and below the middle line.

## 📰 Transformer Architecture
Transformers are neural network architectures that excel in handling sequential data by using self-attention mechanisms to weigh the importance of each element in the sequence. Unlike traditional models, transformers process input in parallel, allowing for faster training and better handling of long-range dependencies.  


LSTM's might struggle for long range dependencies. They still face challenges when it comes to learning relationships across very distant time steps.  

Unlike traditional recurrent neural networks (RNNs), Transformers leverage attention mechanisms to weigh the significance of each input element concerning others. This mechanism allows them to process information in parallel, enhancing efficiency in analyzing complex financial data patterns. This makes transformers particularly powerful for capturing both short-term and long-term dependencies in a sequence  

## 📊 Informer Architecture

The Informer architecture is a time-series forecasting model that leverages the efficient attention mechanism called ProbSparse Self-Attention. It reduces the quadratic complexity of standard attention by focusing on key sparse data points, allowing for faster processing on long sequences. With a combination of encoder-decoder architecture, it improves both speed and accuracy in handling large-scale time-series data.

### ProbSparse Attention mechanism

The self-attention scores form a long-tail distribution, where the "active" queries lie in the "head" scores and "lazy" queries lie in the "tail" area. We designed the ProbSparse Attention to select the "active" queries rather than the "lazy" queries. The ProbSparse Attention with Top-u queries forms a sparse Transformer by the probability distribution. Why not use Top-u keys? The self-attention layer's output is the re-represent of input. It is formulated as a weighted combination of values w.r.t. the score of dot-product pairs. The top queries with full keys encourage a complete re-represent of leading components in the input, and it is equivalent to selecting the "head" scores among all the dot-product pairs. If we choose Top-u keys, the full keys just preserve the trivial sum of values within the "long tail" scores but wreck the leading components' re-represent.

