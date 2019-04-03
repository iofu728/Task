# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-04 19:03:49
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-28 10:26:48

import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import threading
import time

from datetime import datetime
from numba import jit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

data_path = 'concrete/data/'
model_path = 'concrete/model/'
pickle_path = 'concrete/pickle/'
prediction_path = 'concrete/prediction/'
v = '2'
# t = '_total'
t = ''
start = []


def begin_time():
    """
    multi-version time manage
    """
    global start
    start.append(time.time())
    return len(start) - 1


def end_time(version):
    termSpend = time.time() - start[version]
    print(str(termSpend)[0:5])


def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


class Concrete(object):
    """
    data minie for concrete
    """

    def __init__(self, do_pre=False):
        self.id2file = {}
        self.id2lab = {}
        self.detail_map = {}
        self.detail_pickle = {}
        self.f1_max_index = 0.5
        self.f1_map = {index: 0 for index in range(0, 5)}
        self.version = datetime.now().strftime("%m%d%H%M")
        self.seed = 333
        self.EARLY_STOP = 300
        self.OPT_ROUNDS = 2444
        self.MAX_ROUNDS = 300000
        self.evaluate_num = 0

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
        self.pre_data_list(do_pre)

    def load_basic(self, file_type):
        """
        load basic
        @param file_type: 1-submit_example, 0-train_labels
        """

        file_name = 'submit_example' if file_type else 'train_labels'
        file_name += '.csv'
        with open(data_path + file_name, 'r') as f:
            train_list = f.readlines()[1:]
        self.id2file = {
            index: train[:-1].split(',')[0] for index, train in enumerate(train_list)}
        self.id2lab = {index: int(train[:-1].split(',')[1])
                       for index, train in enumerate(train_list)}

    def load_detail(self, file_type, block_size=500):
        """
        load detail
        @param file_type: 1-submit_example, 0-train_labels
        """
        pickle_file = 'submit_middle' if file_type else 'train_middle'
        pickle_file += '.pickle'
        detail_pickle = load_bigger(pickle_path + pickle_file)
        print('load over')
        id_len = len(self.id2lab.keys())
        for block_index in range((id_len - 1) // block_size + 1):
            index_min = block_size * block_index
            index_max = min(id_len, (block_index + 1) * block_size)
            threadings = []
            for index in list(self.id2file.keys())[index_min:index_max]:
                label_id = self.id2lab[index]
                detail_csv = detail_pickle[index]
                work = threading.Thread(
                    target=self.pre_data_once, args=(index, file_type, label_id, detail_csv,))
                threadings.append(work)
            for work in threadings:
                work.start()
            for work in threadings:
                work.join()
            if not index_max % 10:
                print(index_max)
        detail_map = [self.detail_map[k]
                      for k in sorted(self.detail_map.keys())]
        output_file = 'submit_middle' if file_type else 'train_middle'
        title_basic = ['活塞工作时长', '发动机转速', '油泵转速', '泵送压力', '液压油温', '流量档位',
                       '分配压力', '排量电流', '低压开关', '高压开关', '搅拌超压信号', '正泵', '反泵', '设备类型']
        # title_min = [index + '_min'for index in title_basic[1:8]]
        # title_max = [index + '_max'for index in title_basic[1:8]]
        # title_mean = [index + '_mean'for index in title_basic[1:8]]
        # title_std = [index + '_std'for index in title_basic[1:8]]
        # title_poor = [index + '_poor'for index in title_basic[1:8]]
        # title_median = [index + '_median'for index in title_basic[1:8]]
        # title_total = [index + '_total'for index in title_basic[1:8]]
        # title_hit = [index + '_hit'for index in title_basic[1:8]]
        # title_constant = ['label', '活塞工作时长', '低压开关', '正泵', '设备类型', '低压开关&正泵']
        # title_collection = [*title_min, *title_mean, *title_max, *title_poor, *title_std, *title_median, *title_total, *title_hit]
        # title_collection_diff = [index + '_diff' for index in title_collection]
        # title_collection_diff_diff = [
        #     index + '_diff_diff' for index in title_collection]
        # title_collection_diff_diff_diff = [
        #     index + '_diff_diff_diff' for index in title_collection]
        # title_collection_diff_diff_diff2 = [
        #     index + '_diff_diff_diff2' for index in title_collection]
        # title_collection_diff_diff_diff3 = [
        #     index + '_diff_diff_diff3' for index in title_collection]
        # title_collection_ptr = [index + '_pct' for index in title_collection]
        # title_collection_ptr_diff = [
        #     index + '_pct_diff' for index in title_collection]
        # title_all = [*title_constant, *title_collection, *title_collection_diff,
        #              *title_collection_diff_diff, *title_collection_diff_diff_diff,
        #              *title_collection_ptr, *title_collection_ptr_diff,
        #              *title_collection_diff_diff_diff2, *title_collection_diff_diff_diff3]
        # title_all = [*title_collection_diff_diff_diff2, *title_collection_diff_diff_diff3]
        title_skew = [index + '_skew'for index in title_basic[0:8]]
        with open(data_path + output_file, 'w') as f:
            f.write(",".join(title_skew) + '\n')
            # f.write("nunique" + '\n')
            f.write("\n".join([str(index) for index in detail_map]))

    def load_all(self, file_type):
        """
        load all
        """
        self.load_basic(file_type)
        self.load_detail(file_type)
        self.detail_map = {}

    def load_all_pickle(self, file_type):
        """
        load all
        """
        self.load_basic(file_type)
        self.load_detail_pickle(file_type)
        self.detail_pickle = {}

    def load_detail_pickle(self, file_type, block_size=300):
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
                file_id = self.id2file[index]
                work = threading.Thread(
                    target=self.pre_data_two, args=(index, file_type, file_id,))
                threadings.append(work)
            for work in threadings:
                work.start()
            for work in threadings:
                work.join()
            print(index_max)
        output_file = 'submit_middle' if file_type else 'train_middle'
        output_file += '.pickle'
        dump_bigger(self.detail_pickle, pickle_path + output_file)

    def pre_data_list(self, do_pre):
        version = begin_time()
        df_columns = pd.read_csv(data_path + 'train_middle_total').columns
        begin_index = 0

        with open(model_path + v + 'columns.csv', 'r') as f:
            str_f = f.readline()
            if str_f[-1] == '\n':
                str_f = str_f[:-1]
            good_columns = str_f.split(',')

        with open(model_path + v + 'lastcolumn.csv', 'r') as f:
            str_f = f.readline()
            if str_f[-1] == '\n':
                str_f = str_f[:-1]
        while df_columns[begin_index] != str_f:
            begin_index += 1

        self.wait_columns = list(df_columns[begin_index + 6:])
        self.good_columns = good_columns
        self.basic_f1 = 0
        if do_pre == True:
            self.load_all(0)
            self.load_all(1)
        elif do_pre == 2:
            self.load_all_pickle(0)
            self.load_all_pickle(1)
        else:
            self.load_basic(1)
        end_time(version)

    def evaluate_f1(self, preds, train_data):
        self.evaluate_num = self.evaluate_num + 1
        labels = train_data.get_label()
        if not self.evaluate_num % 50:

            f1_list = [self.evaulate_model_once(labels, [int(indexs > (index / 100))
                                                         for indexs in preds]) for index in range(48, 53)]
            max_index = f1_list.index(max(f1_list))
            if max_index in self.f1_map:
                self.f1_map[max_index] += 1
            else:
                self.f1_map[max_index] = 1
            # print(labels, preds)
            return 'f1', f1_list[2], True
        else:
            preds = [int(index > 0.5) for index in preds]
            return 'f1', self.evaulate_model_once(labels, preds), True

    def pre_data_two(self, detail_id, file_type, file_id):
        file_folder = 'data_test' if file_type else 'data_train'
        file_folder += '/'
        file_folder += file_id
        detail_csv = pd.read_csv(data_path + file_folder)
        self.detail_pickle[detail_id] = detail_csv

    def pre_data_once(self, detail_id, file_type, label_id, detail_csv):
        # detail_basic = detail_csv.agg(['min', 'max', 'std', 'mean', 'median'])

        # detail_max = detail_basic.iloc[1]
        # detail_time = detail_csv.max()[0]
        # detail_time = detail_max[0]
        # detail_press = detail_max[8]
        # detail_pump = detail_max[11]
        # detail_type = detail_max[13]
        # detail_add = detail_pump + detail_press
        # detail_constant = [label_id, detail_time,
        #                    detail_press, detail_pump, detail_type, detail_add]

        # detail_max = detail_max[1:8]
        # detail_min = detail_basic.iloc[0, 1:8]
        # detail_poor = detail_max - detail_min
        # detail_mean = detail_basic.iloc[3, 1:8]
        # detail_std = detail_basic.iloc[2, 1:8]
        # detail_median = detail_basic.iloc[4, 1:8]
        # detail_total = [index * detail_time for index in detail_mean]
        # detail_hit = [index * detail_time for index in detail_std]
        # detail_collapse = [*detail_min, *detail_mean, *detail_max, *detail_poor,
        #                    *detail_std, *detail_median, *detail_total, *detail_hit]

        # del detail_csv['设备类型']
        # detail_basic_diff = detail_csv.diff()

        # detail_diff_basic = detail_basic_diff.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_diff_min = detail_diff_basic.iloc[0, 1:8]
        # detail_diff_max = detail_diff_basic.iloc[1, 1:8]
        # detail_diff_poor = detail_diff_max - detail_diff_min
        # detail_diff_std = detail_diff_basic.iloc[2, 1:8]
        # detail_diff_mean = detail_diff_basic.iloc[3, 1:8]
        # detail_diff_median = detail_diff_basic.iloc[4, 1:8]
        # detail_diff_total = [index * detail_time for index in detail_diff_mean]
        # detail_diff_hit = [index * detail_time for index in detail_diff_std]
        # detail_collapse_diff = [*detail_diff_min, *detail_diff_mean, *detail_diff_max, *detail_diff_poor,
        #                         *detail_diff_std, *detail_diff_median, *detail_diff_total, *detail_diff_hit]

        # detail_basic_diff_diff = detail_basic_diff.diff()

        # detail_diff_diff_basic = detail_basic_diff_diff.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_diff_diff_min = detail_diff_diff_basic.iloc[0, 1:8]
        # detail_diff_diff_max = detail_diff_diff_basic.iloc[1, 1:8]
        # detail_diff_diff_poor = detail_diff_diff_max - detail_diff_diff_min
        # detail_diff_diff_std = detail_diff_diff_basic.iloc[2, 1:8]
        # detail_diff_diff_mean = detail_diff_diff_basic.iloc[3, 1:8]
        # detail_diff_diff_median = detail_diff_diff_basic.iloc[4, 1:8]
        # detail_diff_diff_total = [
        #     index * detail_time for index in detail_diff_diff_mean]
        # detail_diff_diff_hit = [
        #     index * detail_time for index in detail_diff_diff_mean]
        # detail_collapse_diff_diff = [*detail_diff_diff_min, *detail_diff_diff_mean, *detail_diff_diff_max,
        #                              *detail_diff_diff_poor, *detail_diff_diff_std, *detail_diff_diff_median,
        #                              *detail_diff_diff_total, *detail_diff_diff_hit]

        # detail_basic_diff_diff_diff = detail_basic_diff_diff.diff()

        # detail_diff_diff_diff_basic = detail_basic_diff_diff_diff.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_diff_diff_diff_min = detail_diff_diff_diff_basic.iloc[0, 1:8]
        # detail_diff_diff_diff_max = detail_diff_diff_diff_basic.iloc[1, 1:8]
        # detail_diff_diff_diff_poor = detail_diff_diff_diff_max - detail_diff_diff_diff_min
        # detail_diff_diff_diff_std = detail_diff_diff_diff_basic.iloc[2, 1:8]
        # detail_diff_diff_diff_mean = detail_diff_diff_diff_basic.iloc[3, 1:8]
        # detail_diff_diff_diff_median = detail_diff_diff_diff_basic.iloc[4, 1:8]
        # detail_diff_diff_diff_total = [
        #     index * detail_time for index in detail_diff_diff_diff_mean]
        # detail_diff_diff_diff_hit = [
        #     index * detail_time for index in detail_diff_diff_diff_mean]
        # detail_collapse_diff_diff_diff = [*detail_diff_diff_diff_min, *detail_diff_diff_diff_mean, *detail_diff_diff_diff_max,
        #                                   *detail_diff_diff_diff_poor, *detail_diff_diff_diff_std, *detail_diff_diff_diff_median,
        #                                   *detail_diff_diff_diff_total, *detail_diff_diff_diff_hit]

        # detail_basic_diff_diff_diff2 = detail_basic_diff_diff_diff.diff()

        # detail_diff_diff_diff2_basic = detail_basic_diff_diff_diff2.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_diff_diff_diff2_min = detail_diff_diff_diff2_basic.iloc[0, 1:8]
        # detail_diff_diff_diff2_max = detail_diff_diff_diff2_basic.iloc[1, 1:8]
        # detail_diff_diff_diff2_poor = detail_diff_diff_diff2_max - detail_diff_diff_diff2_min
        # detail_diff_diff_diff2_std = detail_diff_diff_diff2_basic.iloc[2, 1:8]
        # detail_diff_diff_diff2_mean = detail_diff_diff_diff2_basic.iloc[3, 1:8]
        # detail_diff_diff_diff2_median = detail_diff_diff_diff2_basic.iloc[4, 1:8]
        # detail_diff_diff_diff2_total = [
        #     index * detail_time for index in detail_diff_diff_diff2_mean]
        # detail_diff_diff_diff2_hit = [
        #     index * detail_time for index in detail_diff_diff_diff2_mean]
        # detail_collapse_diff_diff2_diff = [*detail_diff_diff_diff2_min, *detail_diff_diff_diff2_mean, *detail_diff_diff_diff2_max,
        #                                    *detail_diff_diff_diff2_poor, *detail_diff_diff_diff2_std, *detail_diff_diff_diff2_median,
        #                                    *detail_diff_diff_diff2_total, *detail_diff_diff_diff2_hit]

        # detail_basic_diff_diff_diff3 = detail_basic_diff_diff_diff2.diff()

        # detail_diff_diff_diff3_basic = detail_basic_diff_diff_diff3.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_diff_diff_diff3_min = detail_diff_diff_diff3_basic.iloc[0, 1:8]
        # detail_diff_diff_diff3_max = detail_diff_diff_diff3_basic.iloc[1, 1:8]
        # detail_diff_diff_diff3_poor = detail_diff_diff_diff3_max - detail_diff_diff_diff3_min
        # detail_diff_diff_diff3_std = detail_diff_diff_diff3_basic.iloc[2, 1:8]
        # detail_diff_diff_diff3_mean = detail_diff_diff_diff3_basic.iloc[3, 1:8]
        # detail_diff_diff_diff3_median = detail_diff_diff_diff3_basic.iloc[4, 1:8]
        # detail_diff_diff_diff3_total = [
        #     index * detail_time for index in detail_diff_diff_diff3_mean]
        # detail_diff_diff_diff3_hit = [
        #     index * detail_time for index in detail_diff_diff_diff3_mean]
        # detail_collapse_diff_diff3_diff = [*detail_diff_diff_diff3_min, *detail_diff_diff_diff3_mean, *detail_diff_diff_diff3_max,
        #                                    *detail_diff_diff_diff3_poor, *detail_diff_diff_diff3_std, *detail_diff_diff_diff3_median,
        #                                    *detail_diff_diff_diff3_total, *detail_diff_diff_diff3_hit]
        # detail_basic_pct = detail_csv.pct_change()

        # detail_pct_change_basic = detail_basic_pct.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_pct_change_min = detail_pct_change_basic.iloc[0, 1:8]
        # detail_pct_change_max = detail_pct_change_basic.iloc[1, 1:8]
        # detail_pct_change_poor = detail_pct_change_max - detail_pct_change_min
        # detail_pct_change_std = detail_pct_change_basic.iloc[2, 1:8]
        # detail_pct_change_mean = detail_pct_change_basic.iloc[3, 1:8]
        # detail_pct_change_median = detail_pct_change_basic.iloc[4, 1:8]
        # detail_pct_change_total = [
        #     index * detail_time for index in detail_pct_change_mean]
        # detail_pct_change_hit = [
        #     index * detail_time for index in detail_pct_change_std]
        # detail_collapse_ptr = [*detail_pct_change_min, *detail_pct_change_mean, *detail_pct_change_max,
        #                        *detail_pct_change_poor, *detail_pct_change_std, *detail_pct_change_median,
        #                        *detail_pct_change_total, *detail_pct_change_hit]

        # detail_basic_pct_diff = detail_basic_pct.diff()
        # detail_pct_diff_basic = detail_basic_pct_diff.agg(
        #     ['min', 'max', 'std', 'mean', 'median'])
        # detail_pct_diff_min = detail_pct_diff_basic.iloc[0, 1:8]
        # detail_pct_diff_max = detail_pct_diff_basic.iloc[1, 1:8]
        # detail_pct_diff_poor = detail_pct_diff_max - detail_pct_diff_min
        # detail_pct_diff_std = detail_pct_diff_basic.iloc[2, 1:8]
        # detail_pct_diff_mean = detail_pct_diff_basic.iloc[3, 1:8]
        # detail_pct_diff_median = detail_pct_diff_basic.iloc[4, 1:8]
        # detail_pct_diff_total = [
        #     index * detail_time for index in detail_pct_diff_mean]
        # detail_pct_diff_hit = [
        #     index * detail_time for index in detail_pct_diff_std]
        # detail_collapse_pct_diff = [*detail_pct_diff_min, *detail_pct_diff_mean, *detail_pct_diff_max,
        #                             *detail_pct_diff_poor, *detail_pct_diff_std, *detail_pct_diff_median,
        #                             *detail_pct_diff_total, *detail_pct_diff_hit]

        # detail = [*detail_constant, *detail_collapse, *detail_collapse_diff,
        #           *detail_collapse_diff_diff, *detail_collapse_diff_diff_diff,
        #           *detail_collapse_ptr, *detail_collapse_pct_diff,
        #           *detail_collapse_diff_diff2_diff, *detail_collapse_diff_diff3_diff]
        # detail = [*detail_collapse_diff_diff2_diff, *detail_collapse_diff_diff3_diff]

        self.detail_map[detail_id] = ",".join(
            [str(index) for index in list(detail_csv.skew()[0:8])])
        # self.detail_map[detail_id] = ",".join([str(index) for index in detail])
        # self.detail_map[detail_id] = detail_csv['活塞工作时长'].nunique()

    def pre_data(self, pre, slices):
        """
        prepare data
        """
        # detail_type = pd.get_dummies(pre['设备类型'], prefix=pre[['设备类型']].columns[0])
        # pre = pre.drop(['设备类型'], axis=1)
        # return pd.concat([pre, detail_type], axis=1)
        pre['设备类型'] = pre['设备类型'].map(
            {'ZV252': 0, 'ZV573': 1, 'ZV63d': 2, 'ZVa78': 3, 'ZVa9c': 4, 'ZVe44': 4, 'ZVfd4': 5})
        if slices is None:
            return pre
        else:
            columns_total = pre.columns
            if not slices:
                wait_columns = [*columns_total[:128], *columns_total[188:198]]
            if slices == 11:
                wait_columns = [*columns_total[:128]]
            elif slices == 1:
                wait_columns = [*columns_total[:128], *columns_total[178:198]]
            elif slices == 2:
                wait_columns = [*columns_total[:128], *columns_total[178:198], *columns_total[218:228]]
            elif slices < 9:
                wait_columns = [*columns_total[:128], *columns_total[178:198], *columns_total[218:228], *columns_total[88 + slices * 10:98 + slices * 10]]
            else:
                wait_columns = [*columns_total[:128], *columns_total[178:198], *columns_total[218:228], *columns_total[108 + slices * 10:118 + slices * 10]]
            # columns = [*columns_total[:118], *columns_total[178:188], *columns_total[118 + slices * 10:128 + slices * 10]]
            # columns = columns_total[:118] + [:118 + 10 * slices]
            # wait_columns = self.good_columns
            # if slices != -1:
            #     wait_columns = [*wait_columns, self.wait_columns[slices]]
            wait = pd.DataFrame(pre, columns=wait_columns)
            return wait

    def load_data(self, model=True, slices=None):
        """
        load data for appoint model
        @param: model True-train False-predict
        """

        print('Load data...')

        if model:
            pre = pd.read_csv(data_path + 'train_middle' + t)
            target = pre['label'].values
            pre = pre.drop(['label'], axis=1)
            data = self.pre_data(pre, slices)
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.25)
            print('data split end')
            # pre = pd.read_csv('concrete/data/train_middle')
            # target = pre['label'].values
            # pre = pre.drop(['label'], axis=1)
            # X_train = self.pre_data(pre)
            # y_train = target
            # # X_train = self.X_train
            # # y_train = self.y_train

            # pre = pd.read_csv('concrete/data/submit_middle')
            # target = pre['label'].values
            # pre = pre.drop(['label'], axis=1)
            # X_test = self.pre_data(pre)
            # y_test = target
            # print('data split end')
        else:
            pre = pd.read_csv(data_path + 'train_middle' + t)
            target = pre['label'].values
            pre = pre.drop(['label'], axis=1)
            X_train = self.pre_data(pre, slices)
            y_train = target
            # X_train = self.X_train
            # y_train = self.y_train

            pre = pd.read_csv(data_path + 'submit_middle' + t)
            target = pre['label'].values
            pre = pre.drop(['label'], axis=1)
            X_test = self.pre_data(pre, slices)
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

        categorical = ['设备类型']

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
                          feval=self.evaluate_f1)

        importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})

        importances.sort_values('importances', ascending=False, inplace=True)

        model.save_model('concrete/model/{}.model'.format(self.version))
        importances.to_csv(
            'concrete/model/{}_importances.csv'.format(self.version), index=False)

        self.gbm = model
        self.dtrain = dtrain

        # gbm = lgb.LGBMRegressor(
        #     objective='regression', num_leaves=31, learning_rate=0.095, n_estimators=29)
        # gbm.fit(self.X_train, self.y_train, eval_set=[
        #     (self.X_test, self.y_test)], eval_metric='l1', early_stopping_rounds=5)
        # self.gbm = gbm

    def evaulate_model(self, model=True, slices=None):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')

        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration)

        print(self.f1_max_index)
        predict = [int(index > self.f1_max_index) for index in y_pred]
        if model:
            print(self.evaulate_model_once(self.y_test, predict))
            for index in range(30, 70):
                distinguish = index / 100
                predict = [int(index > distinguish) for index in y_pred]
                print(self.evaulate_model_once(self.y_test, predict))
        else:
            file_name = pd.DataFrame(list(self.id2file.values()))
            result = pd.DataFrame({'sample_file_name': file_name[0], 'label': predict}, columns=[
                'sample_file_name', 'label'])
            result.to_csv(
                prediction_path + '{}.csv'.format(self.version + str(slices)), index=False)

    def evaulate_model_once(self, result, predict):
        """
        print evaulate
        """
        result_f1 = (self.F1(result, predict, 0) +
                     self.F1(result, predict, 1)) / 2
        # print(result_f1)
        return result_f1

    @jit
    def F1(self, result, predict, true_value):
        """
        F1
        """
        true_num = 0
        recall_num = 0
        precision_num = 0
        for index, values in enumerate(result):
            # print(index, values, predict[index])
            if values == true_value:
                recall_num += 1
                if values == predict[index]:
                    true_num += 1
            if predict[index] == true_value:
                precision_num += 1
        # print(true_num, recall_num, precision_num)

        R = true_num / recall_num if recall_num else 0
        P = true_num / precision_num if precision_num else 0
        return (2 * P * R) / (P + R) if (P + R) else 0

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
                             categorical_feature=['设备类型'])

        eval_hist = lgb.cv(self.params,
                           dtrain,
                           nfold=5,
                           num_boost_round=self.MAX_ROUNDS,
                           early_stopping_rounds=self.EARLY_STOP,
                           verbose_eval=50,
                           seed=self.seed,
                           shuffle=True,
                           feval=self.evaluate_f1,
                           metrics="None"
                           )
        result = [self.version, self.X_train.columns[-1]]
        result.append('best n_estimators:' + str(len(eval_hist['f1-mean'])))
        result.append('best cv score:' + str(eval_hist['f1-mean'][-1]) + '\n')
        with open(model_path + v + 'result', 'a') as f:
            f.write('\n'.join([str(index) for index in result]))
        print('best n_estimators:', len(eval_hist['f1-mean']))
        print('best cv score:', eval_hist['f1-mean'][-1])
        self.OPT_ROUNDS = len(eval_hist['f1-mean'])
        f1_max_list = [self.f1_map[k] for k in sorted(self.f1_map.keys())]
        print(self.f1_map)
        f1_max_index = range(48, 53)[f1_max_list.index(max(f1_max_list))] / 100
        self.f1_max_index = f1_max_index
        if (eval_hist['f1-mean'][-1] > self.basic_f1):
            self.basic_f1 = eval_hist['f1-mean'][-1]
            if not index is None and index != -1:
                self.good_columns.append(self.wait_columns[index])
        with open(model_path + v + 'columns.csv', 'w') as f:
            f.write(','.join([str(index) for index in self.good_columns]))

        with open(model_path + v + 'lastcolumn.csv', 'w') as f:
            f.write(str(self.X_train.columns[-1]))

        # estimator = lgb.LGBMRegressor(num_leaves=31)

        # param_grid = {
        #     'learning_rate': [0.08, 0.085, 0.09, 0.095, 0.1],
        #     'n_estimators': range()
        # }

        # gbm = GridSearchCV(estimator, param_grid,
        #                    scoring='roc_auc', cv=5, n_jobs=20)

        # gbm.fit(self.X_train, self.y_train)

        # print('Best parameters found by grid search are:', gbm.best_params_)
    def load_one_table(self, file_type):
        """
        load one table
        @param file_type: 1-submit_example, 0-train_labels
        """
        pickle_file = 'submit_middle' if file_type else 'train_middle'
        pickle_file += '.pickle'
        detail_pickle = load_bigger(pickle_path + pickle_file)
        print('load over')

    def test_f1(self, num, dimension):
        """
        random test f1
        """

        result_list = [np.random.randint(0, 2, dimension)
                       for index in range(num)]
        prediction_list = [np.random.randint(
            0, 2, dimension) for index in range(num)]
        version = begin_time()
        for index in range(num):
            self.F1(result_list[index], prediction_list[index], 0)
            self.F1(result_list[index], prediction_list[index], 1)
        end_time(version)

        version = begin_time()
        for index in range(num):
            f1_score(result_list[index], prediction_list[index], pos_label=0)
            f1_score(result_list[index], prediction_list[index], pos_label=1)
        end_time(version)


def empty():
    for ii, jj in test.items():
        temp_max = max(jj.keys())
        for kk in range(1, temp_max):
            if kk not in jj:
                print(ii, kk)
                test[ii][kk] = {}


if __name__ == '__main__':
    version = begin_time()

    model = False
    concrete = Concrete()

    for index in range(12):
        # for index in range(-1, len(concrete.wait_columns)):
        concrete.load_data(model, index)
        concrete.optimize_model(model, index)
        concrete.train_model()
        concrete.evaulate_model(model, index)
    # model = False
    # concrete.load_data(model)
    # concrete.train_model()
    # concrete.evaulate_model(model)
    end_time(version)
