# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-04 19:03:49
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-05 14:52:40

import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import threading

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

from utils.utils import begin_time, end_time


class Concrete(object):
    """
    data minie for concrete
    """

    def __init__(self, do_pre=False):
        self.id2file = {}
        self.id2lab = {}
        self.detail_map = {}
        self.pre_data_list(do_pre)

    def load_basic(self, file_type):
        """
        load basic
        @param file_type: 1-submit_example, 0-train_labels
        """

        file_name = 'submit_example' if file_type else 'train_labels'
        file_name += '.csv'
        with open('concrete/data/' + file_name, 'r') as f:
            train_list = f.readlines()[1:]
        self.id2file = {
            index: train[:-1].split(',')[0] for index, train in enumerate(train_list)}
        self.id2lab = {index: int(train[:-1].split(',')[1])
                       for index, train in enumerate(train_list)}

    def load_detail(self, file_type, block_size=2000):
        """
        load detail
        @param file_type: 1-submit_example, 0-train_labels
        """
        id_len = len(self.id2lab.keys())
        for block_index in range((id_len - 1) // block_size + 1):
            index_min = block_size * block_index
            index_max = min(id_len, (block_index + 1) * block_size)
            threadings = []
            for index in list(self.id2file.keys())[index_min:index_max]:
                work = threading.Thread(
                    target=self.pre_data_once, args=(index, file_type,))
                threadings.append(work)
            for work in threadings:
                work.start()
            for work in threadings:
                work.join()
        detail_map = [self.detail_map[k]
                      for k in sorted(self.detail_map.keys())]
        output_file = 'submit_middle' if file_type else 'train_middle'
        with open('concrete/data/' + output_file, 'w') as f:
            f, write(",".join([str(index)
                               for index in range(len(detail_map[0].split(',')))]) + '\n')
            f.write("\n".join(detail_map))

    def load_all(self, file_type):
        """
        load all
        """
        self.load_basic(file_type)
        self.load_detail(file_type)

    def pre_data_list(self, do_pre, block_size=2000):
        version = begin_time()
        if do_pre:
            self.load_all(0)
            self.load_all(1)
        else:
            self.load_basic(1)
        end_time(version)

    def pre_data_once(self, detail_id, file_type):
        file_folder = 'data_test' if file_type else 'data_train'
        file_folder += '/'
        file_folder += self.id2file[detail_id]
        detail_csv = pd.read_csv('concrete/data/' + file_folder)
        detail_time = detail_csv.max()[0]
        detail_press = detail_csv.max()[8]
        detail_pump = detail_csv.max()[11]
        detail_type = detail_csv.max()[13]
        detail_constant = [detail_time, detail_press, detail_pump, detail_type]
        detail_max = detail_csv.max()[1:8]
        detail_min = detail_csv.min()[1:8]
        detail_poor = detail_max - detail_min
        detail_mean = detail_csv.mean()[1:8]
        detail_std = detail_csv.std()[1:8]
        detail_median = detail_csv.median()[1:8]

        detail = [*detail_constant, *detail_min, *detail_mean, *detail_max, *detail_poor, *detail_std, *detail_median, self.id2lab[detail_id]]

        self.detail_map[detail_id] = ",".join([str(index) for index in detail])

    def pre_data(self, pre):
        """
        prepare data
        """
        feature1 = pd.get_dummies(pre['15'], prefix=pre[['15']].columns[0])
        feature2 = pd.get_dummies(pre['29'], prefix=pre[['29']].columns[0])
        del pre['15']
        del pre['29']
        del pre['1']
        return pd.concat([pre, feature1, feature2], axis=1)

    def load_data(self, model=True):
        """
        load data for appoint model
        @param: model True-train False-predict
        """

        print('Load data...')

        if model:
            pre = pd.read_csv('concrete/data/train_middle')
            target = pre['30'].values

            del pre['30']

            data = self.pre_data(pre)

            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.3)
            print('data split end')
            # X_train = data
            # y_train = target
        else:
            pre = pd.read_csv('concrete/data/train_middle')
            target = pre['30'].values

            del pre['30']

            X_train = self.pre_data(pre)
            X_test = target

            pre = pd.read_csv('concrete/data/submit_middle')
            target = pre['30'].values

            del pre['30']
            y_train = self.pre_data(pre)
            y_test = target
            print('data split end')

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
        predict = [int(index > 0.52) for index in y_pred]
        if model:
            self.evaulate_model_once(self.y_test, predict)
            for index in range(30, 70):
                distinguish = index / 100
                predict = [int(index > distinguish) for index in y_pred]
                self.evaulate_model_once(self.y_test, predict)
        else:
            result = [','.join([self.id2file[index], str(value)])
                      for index, value in enumerate(y_pred)]
            with opne('concrete/data/result.txt', 'w') as f:
                f.write('\n'.join(result))

    def evaulate_model_once(self, result, predict):
        """
        print evaulate
        """
        print((self.F1(result, predict, 0) + self.F1(result, predict, 1)) / 2)

    def F1(self, result, predict, true_value):
        """
        F1
        """
        R = self.recall(result, predict, true_value)
        P = self.precision(result, predict, true_value)
        return (2 * P * R) / (P + R)

    def recall(self, result, predict, true_value):
        """
        recall
        """
        true_num = sum([1 for index, values in enumerate(
            result) if values == true_value and values == predict[index]])
        recall_num = sum([1 for index in result if index == true_value])
        return true_num / recall_num

    def precision(self, result, predict, true_value):
        """
        precision
        """
        true_num = sum([1 for index, values in enumerate(
            result) if values == true_value and values == predict[index]])
        precision_num = sum([1 for index in predict if index == true_value])
        return true_num / precision_num

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
    concrete = Concrete()
    concrete.load_data(model)
    concrete.train_model()
    concrete.evaulate_model(model)
    # elo.optimize_model()
    end_time(version)
