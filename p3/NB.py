#implement NB(Naive Bayes) classifier

import sys
import pandas as pd
import numpy as np
from statistics import NormalDist

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
c_probility = {}
for i in class_list:
    total_dict[i] = []
    temp_df = df[df[len(type)] == str(i)]
    c_probility[i] = len(temp_df) / len(df)
    for j in range(len(type)):
        #categorical features
        if type[j] == 0:
            assert(unique_list[j] is not None)
            _dict = {}
            if_zero_value = False
            for k in unique_list[j]:
                _dict[k] = len(temp_df[temp_df[j] == k])
                if _dict[k] == 0:
                    if_zero_value = True
            #avoid zero possibility
            if if_zero_value:
                for k in unique_list[j]:
                    _dict[k] += 1
            #calculate conditional possibility
            sum = 0
            for k in unique_list[j]:
                sum += _dict[k]
            for k in unique_list[j]:
                _dict[k] /= sum
            total_dict[i].append(_dict)
        #numerical features
        else:
            total_dict[i].append({
                'mean': temp_df[j].mean(), 
                'std': temp_df[j].std()
            })

#testing
df_test = pd.read_csv(sys.argv[2], header=None)
df_test = df_test.drop([len(type)], axis=1)
result = ['class']


for i in range(1, len(df_test)):
    x = df_test.iloc[i].to_numpy()
    predict = {}
    for j in class_list:
        possibility = c_probility[j]
        for val, index in zip(x, range(len(x))):
            #categorical features
            if type[index] == 0:
                possibility *= total_dict[j][index][val]
            else:
                possibility *= NormalDist(mu=total_dict[j][index]['mean'], \
                    sigma=total_dict[j][index]['std']).pdf(val)
        predict[j] = possibility

    result.append(max(predict, key=predict.get))

df_test[len(type)] = result
df_test.to_csv(sys.argv[3], header=False, index=False)