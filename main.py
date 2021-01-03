import os
import pickle
import logging
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from retrieve_data import get_df_from_symbol
from visualize_data import plot_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model


class DataGenerator:

    # constructor
    def __init__(self, data, num_unroll, batch_size=None):
        self._prices = data[["high", "low", "open", "close"]]
        self._prices_length = self._prices.shape[0] - num_unroll
        
        # self._volume = data["volume"]
        
        self._num_unroll = num_unroll

        if not batch_size:
            self._batch_size = self._prices_length - self._num_unroll
        else:
            self._batch_size = batch_size if batch_size <= self._prices_length - self._num_unroll else self._prices_length - self._num_unroll
        
        self._cursor = self._num_unroll
        
        self._empty = False

    # reset data
    def set_data(self, data):
        self._prices = data[["high", "low", "open", "close"]]
        self._prices_length = self._prices.shape[0] - num_unroll
        
        self._volume = data["volume"]
        
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        
        self._cursor = self._num_unroll
        
        self._empty = False

    # get next batch
    def next_batch(self):
        if self._cursor >= self._prices_length: self._empty = True
        batch_X = []
        batch_y = []
        
        for i in range(self._batch_size):
            X = self._prices.iloc[self._cursor - self._num_unroll:self._cursor, :].values
            y = self._prices.iloc[self._cursor, :].values
            
            batch_X.append(X)
            batch_y.append(y)
            
            self._cursor += 1
        
        return np.asarray(batch_X), np.asarray(batch_y)

    # scale data to [0, 1]
    def scale_data(self):
        scaler = MinMaxScaler()
        self._prices = pd.DataFrame(scaler.fit_transform(self._prices), columns = self._prices.columns)
        # self._volume = pd.DataFrame(scaler.fit_transform(self._volume), columns = self._volume.columns)

    # check if empty
    def empty(self):
        return self._empty


# log to console
def console_log(message):
    print("[{0}] {1}".format(datetime.datetime.now().strftime("%d/%b/%Y %H:%M:%S"), message))


# get database connection
def get_connection():
    conn = mysql.connector.connect(user='stock_web', password='test123',
                            host='db', port=3306,
                            database='stock_db')
    return conn


# function to get data from db
def get_symbols():
    conn = get_connection()
    cursor = conn.cursor()
    symbols_df = pd.read_sql_query("SELECT DISTINCT symbol FROM one_min", conn)
    return symbols_df["symbol"].to_list()


def get_symbol_data_from_db(symbol):
    conn = get_connection()
    symbol_data = pd.read_sql_query("SELECT * FROM one_min WHERE symbol = '{0}'".format(symbol), conn);
    symbol_data["date"] = pd.to_datetime(symbol_data["date"], format="%Y-%m-%d %H:%M:%S")
    return symbol_data


# set tensorflow loglevel
def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def get_model(input_shape):
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def test_2(symbol):
    symbol = symbol
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # pickle data to minimize api requests
    data = get_df_from_symbol(symbol)
    # file_name = "{0}--{1}.pickle".format(symbol, datetime.datetime.now().strftime("%Y-%m-%d"))
    # pickle.dump(data, open("data/{0}".format(file_name), "wb"))

    #data = pickle.load(open(f"data/{symbol}--{current_date}.pickle", "rb"))

    dg = DataGenerator(data, num_unroll=10)
    dg.scale_data()

    batch_X = None
    batch_y = None

    batch_X, batch_y = dg.next_batch()
    
    # print(batch_X.shape)
    # print(batch_y.shape)

    model_name = "{0}--{1}.model".format(symbol, datetime.datetime.now().strftime("%Y-%m-%d"))
    # create model
    model = get_model(input_shape=batch_X.shape[1:])

    # train model
    set_tf_loglevel(logging.INFO)
    model.fit(batch_X, batch_y, batch_size=32, epochs=3)

    # save model
    #pickle.dump(model, open("models/{0}".format(model_name), "wb"))

    # load model
    # model = load_model("models/{0}".format(model_name))


if __name__ == "__main__":
    test_2("ABNB")