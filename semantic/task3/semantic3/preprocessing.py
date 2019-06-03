# -*- coding: utf-8 -*- 
# Author: Shengqiang Zhang
# Time  : 2019/5/18 22:04

from argparse import Namespace
from pathlib import Path

import pandas as pd

args = Namespace(
    # source_data_path="data/MSParS.dev",
    train_data_path="data/MSParS.train",
    dev_data_path="data/MSParS.dev",
    test_data_path="data/MSParS.test",
    output_data_path="data/out_MSparS.csv",
    seed=1337
)


def gen_data(path, is_test_data=False):
    data = []
    with Path(path).open(mode="r", encoding="utf-8") as fp:
        lines = fp.readlines()

    id = 0
    for i in range(0, len(lines), 5):
        id += 1
        question_symbol = f"<question id={id}>"
        logical_form_symbol = f"<logical form id={id}>"
        question_to_logical = {}
        for line in lines[i:i + 4]:
            line = line.strip()

            if line.startswith(question_symbol):
                question = line.split(f"{id}>")[1].strip()
                question_to_logical["question"] = question.split()
            if line.startswith(logical_form_symbol):
                if is_test_data:
                    question_to_logical["logical_form"] = None
                else:
                    logical_form = line.split(f"{id}>")[1].strip()
                    question_to_logical["logical_form"] = logical_form.split()
                break
        data.append(question_to_logical)
    return data, len(data)


data = []
train_data, n_train = gen_data(args.train_data_path)
dev_data, n_dev = gen_data(args.dev_data_path)
test_data, n_test = gen_data(args.test_data_path, is_test_data=True)
data.extend(train_data)
data.extend(dev_data)
data.extend(test_data)

# np.random.shuffle(data)
# data_size = len(data)
# n_train = int(data_size * args.perc_train)
# n_dev = int(data_size * args.perc_dev)

for item in data[:n_train]:
    item["split"] = "train"
for item in data[n_train: n_train + n_dev]:
    item["split"] = "dev"
for item in data[n_train + n_dev:]:
    item["split"] = "test"

for item in data[:n_train + n_dev]:
    item["source_language"] = " ".join(item.pop("question"))
    item["target_language"] = " ".join(item.pop("logical_form"))
for item in data[n_train + n_dev:]:
    item["source_language"] = " ".join(item.pop("question"))
    item["target_language"] = " "

data_df = pd.DataFrame(data)
data_df.drop(["logical_form"], axis=1, inplace=True)
data_df.to_csv(args.output_data_path)
