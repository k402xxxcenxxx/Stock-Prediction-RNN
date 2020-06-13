import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import functools

from urllib.request import urlopen
from urllib.error import HTTPError
import json
import time

class Trainer(object):
    def __init__(self, stock_id, input_columns=[3,5,4,6], label_columns=[6], data_folder="data", model_name='', batch_size=250, epochs=500, validation_split=0.1, input_days=5, predict_n_day_after=3):
        self.stock_id = stock_id
        self.input_columns = input_columns
        self.label_columns = label_columns
        self.data_folder = data_folder
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.input_days = input_days
        self.predict_n_day_after = predict_n_day_after

    def prepare_training_data(self):
        full_data = pd.read_csv('%s/%s.csv' % (self.data_folder, self.stock_id), header=None)

        # clean ditry data
        dirty_mask = functools.reduce((lambda x, y: x | y), [full_data[n]=='--' for n in self.input_columns])
        full_data.drop(full_data.loc[dirty_mask])
        dirty_mask = functools.reduce((lambda x, y: x | y), [full_data[n]=='--' for n in self.label_columns])
        full_data.drop(full_data.loc[dirty_mask])

        feature_data = full_data[self.input_columns]
        label_data = full_data[self.label_columns]

        # shift to input_days
        feature_data = functools.reduce((lambda x, y: pd.concat([x, y], axis=1)), [feature_data.shift(-1*n) for n in range(self.input_days)])

        # preparing label data (推移)
        label_data = label_data.shift(-1*self.input_days - self.predict_n_day_after)

        # adjusting the shape of both (去尾)
        [feature_data.drop(feature_data.index[len(feature_data)-1], axis=0, inplace=True) for n in range(self.input_days + self.predict_n_day_after)]
        [label_data.drop(label_data.index[len(label_data)-1], axis=0, inplace=True) for n in range(self.input_days + self.predict_n_day_after)]

        # conversion to numpy array
        x, y = feature_data.values, label_data.values

        # scaling values for model
        self.x_scale = MinMaxScaler()
        self.y_scale = MinMaxScaler()

        self.x_scale.fit(x)
        self.y_scale.fit(y.reshape(-1,1))

        X = self.x_scale.transform(x)
        Y = self.y_scale.transform(y.reshape(-1,len(self.label_columns)))

        # splitting train and test
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.33)
        self.X_train = X_train.reshape((-1,1, len(self.input_columns) * self.input_days))
        self.X_test = X_test.reshape((-1,1, len(self.input_columns) * self.input_days))

        self.feature_data = feature_data
        self.label_data = label_data
        self.x = x
        self.y = y

    def create_model(self):
        model = Sequential()
        model.add(GRU(units=512,
                    return_sequences=True,
                    input_shape=(1, len(self.input_columns) * self.input_days)))
        model.add(Dropout(0.2))
        model.add(GRU(units=256))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.label_columns), activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam')

        self.model = model

    def train(self):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split, verbose=1)
        self.model.save("{}.h5".format(self.model_name))
        print('MODEL-SAVED')

    def get_score(self):
        score = self.model.evaluate(self.X_test, self.y_test)
        print('Score: {}'.format(score))

    def plot_result(self):
        yhat = self.model.predict(self.X_test)
        yhat = self.y_scale.inverse_transform(yhat)
        self.y_test = self.y_scale.inverse_transform(self.y_test)
        plt.plot(yhat[-100:], label='Predicted')
        plt.plot(self.y_test[-100:], label='Ground Truth')
        plt.legend()
        plt.show()



    def verify_to_current_data(self, date_str='20190601'):

        api_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=%s&stockNo=%s&_=%s"
        timestamp = time.time()

        #　query data
        query_url = api_url % (date_str, self.stock_id, timestamp)

        try:
            result = urlopen(query_url)
        except HTTPError as e:
            print(e.code)
            print(e.read())

            return

        result = json.loads(result.read())

        if(result['stat'] != 'OK'):
            raise Exception("query fail, query_url=%s"%query_url)

        real_data = pd.DataFrame(result['data'])

        real_feature_data = real_data[self.input_columns].astype(float)
        real_label_data = real_data[self.label_columns].astype(float)

        # shift to input_days
        real_feature_data = functools.reduce((lambda x, y: pd.concat([x, y], axis=1)), [real_feature_data.shift(-1*n) for n in range(self.input_days)])

        # preparing label data (推移)
        real_label_data = real_label_data.shift(-1*self.input_days)

        # adjusting the shape of both (去尾)
        [real_feature_data.drop(real_feature_data.index[len(real_feature_data)-1], axis=0, inplace=True) for n in range(self.input_days)]
        [real_label_data.drop(real_label_data.index[len(real_label_data)-1], axis=0, inplace=True) for n in range(self.input_days)]

        # conversion to numpy array
        x, y = real_feature_data.values, real_label_data.values

        x = self.x_scale.transform(x)
        x = x.reshape((-1,1, len(self.input_columns) * self.input_days))

        yhat = self.model.predict(x)
        yhat = self.y_scale.inverse_transform(yhat)
        plt.plot(yhat, label='Predicted')
        plt.plot(y, label='Ground Truth')
        plt.legend()
        plt.show()
