#implement NB(Naive Bayes) classifier

import sys
import pandas as pd
import numpy as np

#read raw training data
df = pd.read_csv(sys.argv[1], header=None)
type = df.iloc[0].to_numpy()[:-1]
df = df.drop([0, 0])
class_list = df[len(type)].unique()

unique_list = []
for i in range(len(type)):
    if type[i] == 0:
        unique_list.append(df[i].unique())
    else:
        unique_list.append(None)

total_dict = {}
for i in class_list:
    total_dict[i] = []
    temp_df = df[df[len(type)] == str(i)]
    for j in range(len(type)):
        if type[j] == 0:
            assert(unique_list[j] is not None)
            _dict = {}
            for k in unique_list[j]:
                _dict[k] = len(temp_df[temp_df[j] == k])
            total_dict[i].append(_dict)
        else:
            total_dict[i].append(None)
for key, value in total_dict.items():
    print(key)
    print(value)
    print()