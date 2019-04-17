# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-01-24 15:41:13
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-04 19:10:32
from util.util import begin_time, end_time
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings

from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import ensemble
from sklearn import model_selection

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


def drop_col_not_req(df, cols):
    df.drop(cols, axis=1, inplace=True)


def find_primenum(uppernum):
    """
    find x in [2, uppernum] which x is a prime number
    """
    assert isinstance(uppernum, (int))
    assert uppernum >= 2
    temp = range(2, uppernum + 1)
    result = []
    while len(temp) > 0:
        result.append(temp[0])
        temp = [index for index in temp[1:] if index % temp[0]]
    return result


class elo_lgb(object):
    """
    train for Elo Merchant Category Recommendation by lightgbm
    """

    def data_pre(self, wait, uppernum=20000, train_len=201917):
        """
        pre data
        """
        wait.first_active_month.fillna('2018-02', inplace=True)
        wait['Year'] = wait.first_active_month.map(
            lambda day: int(day[2:4])).astype('int8')
        wait['Month'] = wait.first_active_month.map(
            lambda day: int(day[5:7])).astype('int8')
        year = pd.get_dummies(
            wait['Year'], prefix=wait[['Year']].columns[0]).astype('int8')
        month = pd.get_dummies(
            wait['Month'], prefix=wait[['Month']].columns[0]).astype('int8')
        del wait['first_active_month']

        wait['ID1'] = wait.card_id.map(lambda ID: int('0x' + ID[6:15], 16))

        # prime_list = find_primenum(uppernum)
        # prime_list = prime_list[-20:]
        # print(prime_list)
        # maxIndex, maxStd = [0, 0]
        # print('prime begin')
        # temp_wait = wait[:train_len]
        # for index in prime_list:
        #     ID2 = temp_wait.ID1.map(lambda ID: ID % index).astype('int16')
        #     pre = pd.concat([ID2, temp_wait.target], axis=1)
        #     temp_std = pre.groupby(by=['ID1']).mean().target.std()
        #     if maxStd < temp_std:
        #         maxIndex = index
        #         maxStd = temp_std
        # print('prime end', maxIndex)
        # wait['ID2'] = wait.ID1.map(lambda ID: ID % maxIndex)
        # ID3 = pd.get_dummies(wait['ID2'], prefix=wait[['ID2']].columns[0])
        del wait['card_id']
        print('ID end')

        feature1 = pd.get_dummies(
            wait['feature_1'], prefix=wait[['feature_1']].columns[0]).astype('int8')
        feature2 = pd.get_dummies(
            wait['feature_2'], prefix=wait[['feature_2']].columns[0]).astype('int8')
        feature3 = pd.get_dummies(
            wait['feature_3'], prefix=wait[['feature_3']].columns[0]).astype('int8')

        del wait['target']
        print('feature end')

        test = pd.concat([wait, year, month, feature1,
                          feature2, feature3], axis=1)
        print('wait begin')
        # wait = ID3
        print('copy end')
        for index in test.axes[1]:
            wait[index] = test[index]
        print('data pre end')
        return wait.values

    def load_data(self, model=True):
        """
        load data for appoint model
        @param: model True-train False-predict
        """

        print('Load data...')

        if model:
            pre = pd.read_csv('elo/data/train.csv')
            target = pre.target.values
            data = self.data_pre(pre, train_len=len(target))

            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2)
            print('data split end')
            # X_train = data
            # y_train = target
        else:
            pre = pd.read_csv('elo/data/train.csv')
            y_train = pre.target.values
            train_len = len(y_train)
            temp_test = pd.read_csv('elo/data/test.csv')

            y_test = pd.read_csv(
                'elo/data/sample_submission.csv').target.values
            temp_test.target = y_test
            # return self.data_pre(
            #     pd.concat([pre, temp_test], axis=0), train_len)
            total = self.data_pre(
                pd.concat([pre, temp_test], axis=0), train_len=train_len)
            X_train = total[:train_len]
            X_test = total[train_len:]
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train

    def train_model(self):
        """
        train model by lightgbm
        """

        print('Start training...')
        gbm = lgb.LGBMRegressor(
            objective='regression', num_leaves=31, learning_rate=0.095, n_estimators=29)
        gbm.fit(self.X_train, self.y_train, eval_set=[
            (self.X_test, self.y_test)], eval_metric='l1', early_stopping_rounds=5)
        self.gbm = gbm

    def evaulate_model(self, model=True):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')
        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration_)
        result = pd.DataFrame(y_pred - self.y_test).std()
        print(result)

        if not model:
            pre = pd.read_csv('elo/data/sample_submission.csv')
            pre.target = y_pred
            pre.to_csv('elo/data/result.csv', index=False)

    def optimize_model(self):
        """
        optimize model by lightgbm
        """
        print('Feature importances:', list(self.gbm.feature_importances_))

        estimator = lgb.LGBMRegressor(num_leaves=31)

        param_grid = {
            'learning_rate': [0.08, 0.085, 0.09, 0.095, 0.1],
            'n_estimators': [25, 26, 27, 28, 29, 30]
        }

        gbm = GridSearchCV(estimator, param_grid)

        gbm.fit(self.X_train, self.y_train)

        print('Best parameters found by grid search are:', gbm.best_params_)


if __name__ == '__main__':
    version = begin_time()
    model = False
    elo = elo_lgb()
    elo.load_data(model)
    elo.train_model()
    elo.evaulate_model(model)
    # elo.optimize_model()
    end_time(version)
