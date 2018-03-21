import csv
import os
import numpy as np

'''将DT.train.csv中的数据分成a_train和a_test两个csv文件，其中a_train.csv中有24500个数据，a_test.csv中有10500个数据'''
"""将DT.TrainValue.csv数据分成train_value和test_value两个文件，其中train_value有27000个数据，test_value有11691个数据"""
labels = []
data = []
a_train_file = 'a_train.csv'
a_test_file = 'a_test.csv'
a_file = 'DT.TrainValue.csv'

seed = 3
np.random.seed(seed)
train_indices = np.random.choice(38691, 27000, replace=False)
residue = np.array(list(set(range(38691)) - set(train_indices)))
# print(np.array(train_indices).shape)  # (27000,)
# print(residue.shape)  # (10944,)
test_indices = np.random.choice(len(residue), 11691, replace=False)
# print(np.array(test_indices).shape)  # (10500,)

with open(a_file)as afile:
    a_reader = csv.reader(afile)
    labels = next(a_reader)
    for row in a_reader:
        data.append(row)

# print(np.array(labels).shape)  # (24,)
# print(np.array(data).shape)  # (35444, 24))


# 生成训练数据集
if not os.path.exists(a_train_file):
    with open(a_train_file, "w", newline='') as a_trian:
        writer = csv.writer(a_trian)
        writer.writerows([labels])
        writer.writerows(np.array(data)[train_indices])
        a_trian.close()

# 生成测试数据集
if not os.path.exists(a_test_file):
    with open(a_test_file, "w", newline='')as a_test:
        writer = csv.writer(a_test)
        writer.writerows([labels])
        writer.writerows(np.array(data)[test_indices])
        a_test.close()
