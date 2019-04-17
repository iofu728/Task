# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-18 15:14:48
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-19 12:58:55
import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import threading
import re

from datetime import datetime
from numba import jit
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

from util.util import begin_time, end_time, dump_bigger, load_bigger

data_path = 'interview/data/'
model_path = 'interview/model/'
pickle_path = 'interview/pickle/'
prediction_path = 'interview/prediction/'


class SA(object):
    """
    data minie for classify
    """

    def __init__(self, do_pre=False):
        self.version = datetime.now().strftime("%m%d%H%M")
        self.seed = 333
        self.EARLY_STOP = 300
        self.OPT_ROUNDS = 2444
        self.MAX_ROUNDS = 300000
        self.evaluate_num = 0
        self.basic_auc = 0

        self.params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.01,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.85,
            'subsample_freq': 10,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'num_leaves': 63,
            'seed': self.seed,
            'nthread': 20,
            'metric': "None",
            "verbose": -1
        }
        self.load_data_constant()

    def pre_data_list(self, do_pre):
        version = begin_time()
        if do_pre == True:
            self.load_all(0)
            self.load_all(1)
        elif do_pre == 2:
            self.load_all_pickle(0)
            self.load_all_pickle(1)
        else:
            self.load_basic(1)
        end_time(version)

    @jit
    def fast_auc(self, y_true, y_prob):
        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        nfalse = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            nfalse += (1 - y_i)
            auc += y_i * nfalse
        auc /= (nfalse * (n - nfalse))
        return auc

    def eval_auc(self, preds, dtrain):
        labels = dtrain.get_label()
        return 'auc', self.fast_auc(labels, preds), True

    def load_data_constant(self):
        """
        load data constant coloums
        """
        df = pd.read_csv(data_path + 'train.csv')
        df_nunique = df.nunique()
        df_columns = [
            index for index in df_nunique.index if df_nunique[index] != 1]
        df = pd.DataFrame(df, columns=df_columns)

        df_02 = df.quantile(0.2)
        df_08 = df.quantile(0.8)
        middle_constant_columns = [
            index for index in df_08.index if df_02[index] == df_08[index]][:-1]
        middle_value = pd.DataFrame(
            df, columns=middle_constant_columns).quantile(0.5)
        basic_columns = [
            index for index in df.columns if index not in middle_constant_columns]
        var_columns = [re.findall('var[0-9]*', index)[0] if len(
            re.findall('var[0-9]*', index)) else 1 for index in df_columns]
        var_map = {index: [] for index in set(var_columns)}
        for index, var in enumerate(var_columns):
            var_map[var].append(df_columns[index])
        del var_map[1]

        underline_1 = ['underline_1' if index.count(
            '_') == 1 else 1 for index in df_columns]
        underline_1_map = {index: [] for index in set(underline_1)}
        for index, var in enumerate(underline_1):
            underline_1_map[var].append(df_columns[index])
        del underline_1_map[1]

        underline_2 = ['underline_2' if index.count(
            '_') == 2 else 1 for index in df_columns]
        underline_2_map = {index: [] for index in set(underline_2)}
        for index, var in enumerate(underline_2):
            underline_2_map[var].append(df_columns[index])
        del underline_2_map[1]

        underline_3 = ['underline_3' if index.count(
            '_') == 3 else 1 for index in df_columns]
        underline_3_map = {index: [] for index in set(underline_3)}
        for index, var in enumerate(underline_3):
            underline_3_map[var].append(df_columns[index])
        del underline_3_map[1]

        underline_4 = ['underline_4' if index.count(
            '_') == 4 else 1 for index in df_columns]
        underline_4_map = {index: [] for index in set(underline_4)}
        for index, var in enumerate(underline_4):
            underline_4_map[var].append(df_columns[index])
        del underline_4_map[1]

        underline_5 = ['underline_5' if index.count(
            '_') == 5 else 1 for index in df_columns]
        underline_5_map = {index: [] for index in set(underline_5)}
        for index, var in enumerate(underline_5):
            underline_5_map[var].append(df_columns[index])
        del underline_5_map[1]

        underline_1_0 = [index.split('_')[0] if index.count(
            '_') == 1 else 1 for index in df_columns]
        underline_1_0_map = {index: [] for index in set(underline_1_0)}
        for index, var in enumerate(underline_1_0):
            underline_1_0_map[var].append(df_columns[index])
        del underline_1_0_map[1]

        underline_1_1 = [index.split('_')[-1] if index.count(
            '_') == 1 else 1 for index in df_columns]
        underline_1_1_map = {index: [] for index in set(underline_1_1)}
        for index, var in enumerate(underline_1_1):
            underline_1_1_map[var].append(df_columns[index])
        del underline_1_1_map[1]

        underline_2_0 = [index.split('_')[0] if index.count(
            '_') == 2 else 1 for index in df_columns]
        underline_2_0_map = {index: [] for index in set(underline_2_0)}
        for index, var in enumerate(underline_2_0):
            underline_2_0_map[var].append(df_columns[index])
        del underline_2_0_map[1]

        underline_2_1 = [index.split('_')[1] if index.count(
            '_') == 2 else 1 for index in df_columns]
        underline_2_1_map = {index: [] for index in set(underline_2_1)}
        for index, var in enumerate(underline_2_1):
            underline_2_1_map[var].append(df_columns[index])
        del underline_2_1_map[1]

        underline_2_2 = [index.split('_')[2] if index.count(
            '_') == 2 else 1 for index in df_columns]
        underline_2_2_map = {index: [] for index in set(underline_2_2)}
        for index, var in enumerate(underline_2_2):
            underline_2_2_map[var].append(df_columns[index])
        del underline_2_2_map[1]

        underline_3_0 = [(index.split('_')[0] if 'var' in index.split('_')[1] else index.split(
            '_')[0] + '_' + index.split('_')[1]) if index.count('_') == 3 else 1 for index in df_columns]
        underline_3_0_map = {index: [] for index in set(underline_3_0)}
        for index, var in enumerate(underline_3_0):
            underline_3_0_map[var].append(df_columns[index])
        del underline_3_0_map[1]

        underline_3_1 = [(index.split('_')[1] if 'var' in index.split('_')[1] else index.split(
            '_')[2]) if index.count('_') == 3 else 1 for index in df_columns]
        underline_3_1_map = {index: [] for index in set(underline_3_1)}
        for index, var in enumerate(underline_3_1):
            underline_3_1_map[var].append(df_columns[index])
        del underline_3_1_map[1]

        underline_3_2 = [("_".join(index.split('_')[2:4]) if 'var' in index.split('_')[1] else index.split(
            '_')[3]) if index.count('_') == 3 else 1 for index in df_columns]
        underline_3_2_map = {index: [] for index in set(underline_3_2)}
        for index, var in enumerate(underline_3_2):
            underline_3_2_map[var].append(df_columns[index])
        del underline_3_2_map[1]

        underline_4_0 = [("_".join(index.split('_')[0:2]) if 'var' in index.split('_')[2] else "_".join(
            index.split('_')[0:3])) if index.count('_') == 4 else 1 for index in df_columns]
        underline_4_0_map = {index: [] for index in set(underline_4_0)}
        for index, var in enumerate(underline_4_0):
            underline_4_0_map[var].append(df_columns[index])
        del underline_4_0_map[1]

        underline_4_1 = [(index.split('_')[2] if 'var' in index.split('_')[2] else index.split(
            '_')[3]) if index.count('_') == 4 else 1 for index in df_columns]
        underline_4_1_map = {index: [] for index in set(underline_4_1)}
        for index, var in enumerate(underline_4_1):
            underline_4_1_map[var].append(df_columns[index])
        del underline_4_1_map[1]

        underline_4_2 = [("_".join(index.split('_')[3:5]) if 'var' in index.split('_')[2] else index.split(
            '_')[4]) if index.count('_') == 4 else 1 for index in df_columns]
        underline_4_2_map = {index: [] for index in set(underline_4_2)}
        for index, var in enumerate(underline_4_2):
            underline_4_2_map[var].append(df_columns[index])
        del underline_4_2_map[1]

        underline_5_0 = ['_'.join(index.split('_')[0:3]) if index.count(
            '_') == 5 else 1 for index in df_columns]
        underline_5_0_map = {index: [] for index in set(underline_5_0)}
        for index, var in enumerate(underline_5_0):
            underline_5_0_map[var].append(df_columns[index])
        del underline_5_0_map[1]

        underline_5_1 = [index.split('_')[3] if index.count(
            '_') == 5 else 1 for index in df_columns]
        underline_5_1_map = {index: [] for index in set(underline_5_1)}
        for index, var in enumerate(underline_5_1):
            underline_5_1_map[var].append(df_columns[index])
        del underline_5_1_map[1]

        underline_5_2 = ["_".join(index.split('_')[4:6]) if index.count(
            '_') == 5 else 1 for index in df_columns]
        underline_5_2_map = {index: [] for index in set(underline_5_2)}
        for index, var in enumerate(underline_5_2):
            underline_5_2_map[var].append(df_columns[index])
        del underline_5_2_map[1]

        with open(model_path + 'columns.csv', 'r') as f:
            str_f = f.readline()
            if str_f[-1] == '\n':
                str_f = str_f[:-1]
            good_columns = str_f.split(',')

        self.df_columns = df_columns
        self.middle_constant_columns = middle_constant_columns
        self.middle_value = middle_value
        self.basic_columns = basic_columns[:-1][1:]
        self.good_columns = good_columns
        # self.good_columns = ['ID']

        # self.wait_columns = [index for index in df_columns if index not in basic_columns]
        self.wait_map = {
            'var_map': var_map,
            'underline_1_0': underline_1_0_map,
            'underline_1_1': underline_1_1_map,
            'underline_2_0': underline_2_0_map,
            'underline_2_1': underline_2_1_map,
            'underline_2_2': underline_2_2_map,
            'underline_3_0': underline_3_0_map,
            'underline_3_1': underline_3_1_map,
            'underline_3_2': underline_3_2_map,
            'underline_4_0': underline_4_0_map,
            'underline_4_1': underline_4_1_map,
            'underline_4_2': underline_4_2_map,
            'underline_5_0': underline_5_0_map,
            'underline_5_1': underline_5_1_map,
            'underline_5_2': underline_5_2_map,
            'underline_1': underline_1_map,
            'underline_2': underline_2_map,
            'underline_3': underline_3_map,
            'underline_4': underline_4_map,
            'underline_5': underline_5_map,
        }
        self.wait_columns = []
        for idx in self.wait_map.keys():
            temp_sum = [str(index) + '_' + idx +
                        '_sum' for index in self.wait_map[idx]]
            temp_max = [str(index) + '_' + idx +
                        '_max' for index in self.wait_map[idx]]
            temp_min = [str(index) + '_' + idx +
                        '_min' for index in self.wait_map[idx]]
            temp_poor = [str(index) + '_' + idx +
                         '_poor' for index in self.wait_map[idx]]
            temp_medain = [str(index) + '_' + idx +
                           '_medain' for index in self.wait_map[idx]]
            temp_std = [str(index) + '_' + idx +
                        '_std' for index in self.wait_map[idx]]
            self.wait_columns = [*self.wait_columns, *temp_sum, *temp_max, *temp_min, *temp_poor, *temp_medain, *temp_std]

        # middle_list = ['Middle_sum', 'Middle_max', 'Middle_min',
        #                'Middle_poor', 'Middle_median', 'Middle_std']
        # self.wait_columns = [*self.wait_columns, *middle_list]
        # self.wait_columns.append('Middle')
        self.var_map = var_map
        self.wait_columns = good_columns[1:]

    def pre_data_v1(self, types):
        file_type = 'train' if types == 1 else 'test'
        # file_address = data_path + file_type + '.csv'
        # df = pd.read_csv(file_address)
        # df = pd.DataFrame(df, columns=self.df_columns)

        file_address = pickle_path + file_type + '.pickle'
        df = pd.read_pickle(file_address)

        # change dirty data
        if types == 1:
            self.var3_mean = sum([index for index in df['var3']
                                  if index > 0]) / len([1 for index in df['var3'] if index > 0])
        df['var3_dirt_new'] = df['var3'].map(
            lambda x: self.var3_mean if x < 0 else x)
        df['saldo_medio_var12_ult1_new'] = df['saldo_medio_var12_ult1'].map(
            lambda x: x > 3077)

        # std big data
        if types == 1:
            self.delta_imp_aport_var13_1y3_mean = sum([index for index in df['delta_imp_aport_var13_1y3']
                                                       if index < 1000]) / len([1 for index in df['delta_imp_aport_var13_1y3'] if index < 1000])
        df['delta_imp_aport_var13_1y3_mean'] = df['delta_imp_aport_var13_1y3'].map(
            lambda x: self.delta_imp_aport_var13_1y3_mean if x > 1000 else x)

        if types == 1:
            self.delta_imp_compra_var44_1y3_mean = sum([index for index in df['delta_imp_compra_var44_1y3']
                                                        if index < 1000]) / len([1 for index in df['delta_imp_compra_var44_1y3'] if index < 1000])
        df['delta_imp_compra_var44_1y3_mean'] = df['delta_imp_compra_var44_1y3'].map(
            lambda x: self.delta_imp_compra_var44_1y3_mean if x > 1000 else x)

        # max_index = 2
        # max_num = 0
        # for slices in range(2, 10000):
        #     temp_percent = sum([df['TARGET'][index] for index, idx in enumerate(
        #         df['saldo_medio_var12_ult1']) if index > slices]) / sum([1 for index in df['saldo_medio_var12_ult1'] if index > slices])
        #     if (temp_percent > max_num):
        #         max_num = temp_percent
        #         max_index = slices

        # middle
        # df_middle = pd.DataFrame(df, columns=self.middle_constant_columns)
        # df_middle_equal = df_middle == self.middle_value
        # df_middle_another = pd.DataFrame()
        # for index in df_middle_equal.columns:
        #     df_middle_another[index + 'middle'] = df_middle_equal[index]

        # df['Middle_sum'] = df_middle_equal.sum(axis=1)
        # df['Middle_max'] = df_middle_equal.max(axis=1)
        # df['Middle_min'] = df_middle_equal.min(axis=1)
        # df['Middle_std'] = df_middle_equal.std(axis=1)
        # df['Middle_poor'] = df['Middle_max'] - df['Middle_min']
        # df['Middle_median'] = df_middle_equal.median(axis=1)

        # df['Middle1'] = df_middle_equal.sum(axis=1).map(lambda x: bool(x))
        # df = pd.concat([df, df_middle_another], axis=1)
        # for idx in self.wait_map.keys():
        #     for index in self.wait_map[idx].keys():
        #         df[index + '_' + idx + '_sum'] = pd.DataFrame(
        #             df, columns=self.wait_map[idx][index]).sum(axis=1)
        #         df[index + '_' + idx + '_max'] = pd.DataFrame(
        #             df, columns=self.wait_map[idx][index]).max(axis=1)
        #         df[index + '_' + idx + '_min'] = pd.DataFrame(
        #             df, columns=self.wait_map[idx][index]).min(axis=1)
        #         df[index + '_' + idx + '_poor'] = df[index + '_' +
        #                                              idx + '_max'] - df[index + '_' + idx + '_min']
        #         df[index + '_' + idx + '_medain'] = pd.DataFrame(
        #             df, columns=self.wait_map[idx][index]).median(axis=1)
        #         df[index + '_' + idx + '_std'] = pd.DataFrame(
        #             df, columns=self.wait_map[idx][index]).std(axis=1)

        df.to_pickle(pickle_path + file_type + '.pickle')

    def pre_data(self, pre, slices):
        """
        prepare data
        """
        if slices is None:
            # with open(model_path + 'columns.csv', 'r') as f:
            #     wait_columns = f.readline()[:-1].split(',')
            wait_columns = self.good_columns
            pre = pd.DataFrame(pre, columns=wait_columns)
            return pre
        else:
            wait_columns = self.good_columns
            if slices != -1:
                wait_columns = [*wait_columns, self.wait_columns[slices]]
            wait = pd.DataFrame(pre, columns=wait_columns)
            return wait

    def load_data(self, model=True, slices=None):
        """
        load data for appoint model
        @param: model True-train False-predict
        """

        print('Load data...')

        if model:
            pre = pd.read_pickle(pickle_path + 'train.pickle')
            target = pre['TARGET'].values
            pre = pre.drop(['TARGET'], axis=1)
            data = self.pre_data(pre, slices)
            data = pre
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.25)
            print('data split end')
        else:
            pre = pd.read_pickle(pickle_path + 'train.pickle')
            target = pre['TARGET'].values
            pre = pre.drop(['TARGET'], axis=1)
            X_train = self.pre_data(pre, slices)
            # X_train = pre
            y_train = target

            pre = pd.read_pickle(pickle_path + 'test.pickle')
            target = pre['TARGET'].values
            pre = pre.drop(['TARGET'], axis=1)
            X_test = self.pre_data(pre, slices)
            # X_test = pre
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

        categorical = []

        dtrain = lgb.Dataset(self.X_train,
                             label=self.y_train,
                             feature_name=list(self.X_train.columns),
                             categorical_feature=categorical)

        model = lgb.train(self.params,
                          dtrain,
                          num_boost_round=self.OPT_ROUNDS,
                          valid_sets=[dtrain],
                          valid_names=['train'],
                          verbose_eval=100,
                          feval=self.eval_auc)

        importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})

        importances.sort_values('importances', ascending=False, inplace=True)

        model.save_model(model_path + '{}.model'.format(self.version))
        importances.to_csv(
            model_path + '{}_importances.csv'.format(self.version), index=False)

        self.gbm = model
        self.dtrain = dtrain

    def evaulate_model(self, model=True):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')

        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration)

        # print(self.auc_max_index)
        # predict = [int(index > self.auc_max_index) for index in y_pred]
        print(self.fast_auc(self.y_test, y_pred))
        with open(model_path + 'result', 'a') as f:
            f.write(str(self.fast_auc(self.y_test, y_pred)) + '\n')
        if not model:
            result = pd.DataFrame({'sample_file_name': self.X_test.ID, 'label': y_pred}, columns=[
                'sample_file_name', 'label'])
            result.to_csv(prediction_path +
                          '{}.csv'.format(self.version), index=False)

    def optimize_model(self, model, index=None):
        """
        optimize model by lightgbm
        """
        # print('Feature importances:', list(self.gbm.feature_importance()))
        print(self.X_train.iloc[0, ], self.X_train.columns, len(
            self.X_train.columns), self.y_train[0])
        dtrain = lgb.Dataset(self.X_train,
                             label=self.y_train,
                             feature_name=list(self.X_train.columns),
                             categorical_feature=[])

        eval_hist = lgb.cv(self.params,
                           dtrain,
                           nfold=8,
                           num_boost_round=self.MAX_ROUNDS,
                           early_stopping_rounds=self.EARLY_STOP,
                           verbose_eval=50,
                           seed=self.seed,
                           shuffle=True,
                           feval=self.eval_auc,
                           metrics="None"
                           )
        result = [self.version]
        result.append('best n_estimators:' + str(len(eval_hist['auc-mean'])))
        result.append('best cv score:' + str(eval_hist['auc-mean'][-1]) + '\n')
        with open(model_path + 'result', 'a') as f:
            f.write('\n'.join([str(index) for index in result]))
        print('best n_estimators:', len(eval_hist['auc-mean']))
        print('best cv score:', eval_hist['auc-mean'][-1])
        self.OPT_ROUNDS = len(eval_hist['auc-mean'])
        if (eval_hist['auc-mean'][-1] > self.basic_auc):
            self.basic_auc = eval_hist['auc-mean'][-1]
            if not index is None and index != -1:
                self.good_columns.append(self.wait_columns[index])
        with open(model_path + 'columns.csv', 'w') as f:
            f.write(','.join([str(index) for index in self.good_columns]))


if __name__ == '__main__':
    version = begin_time()

    model = False
    single = True
    im = SA()
    # im.pre_data_v1(1)
    # im.pre_data_v1(0)
    # single = True
    if single:
        im.load_data(model)
        im.optimize_model(model)
        im.train_model()
        im.evaulate_model(model)

    else:
        for index in range(-1, len(im.wait_columns)):  # filter good feature
            im.load_data(model, index)
            im.optimize_model(model, index)
            im.train_model()
            im.evaulate_model(not model)
    # im.train_model()
    # im.evaulate_model(model)
    # model = False
    # concrete.load_data(model)
    # concrete.train_model()
    # concrete.evaulate_model(model)
    end_time(version)
