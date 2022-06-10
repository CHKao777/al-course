#implement NB(Naive Bayes) classifier

import sys
import pandas as pd
from statistics import NormalDist

#read raw training data
df = pd.read_csv(sys.argv[1], header=None, dtype=str)
var_type = df.iloc[0].to_numpy()[:-1]
var_type = [int(x) for x in var_type]
df = df.drop([0, 0])

# filling missing datas
for i in range(len(var_type)):
    missing_count = len(df[i][df[i] == ' ?'])
    if missing_count != 0:
        #assign an extra category -1 to missing categorical value 
        if var_type[i] == 0:
            df[i] = df[i].replace(' ?', '-1')
        #assign mean to missing numerical value
        elif var_type[i] == 1:
            mean = df[i][df[i] != ' ?'].astype(float).mean()
            df[i] = df[i].replace(' ?', str(mean))
        else:
            raise RuntimeError('var_type is wrong')

class_list = df[len(var_type)].unique()

var_class_list = []
for i in range(len(var_type)):
    if var_type[i] == 0:
        var_class_list.append(df[i].unique())
    elif var_type[i] == 1:
        var_class_list.append(None)
    else:
        raise RuntimeError('var_type is wrong')

total_conditional_probability_dict = {}
c_probability = {}
for i in class_list:
    total_conditional_probability_dict[i] = []
    temp_df = df[df[len(var_type)] == i]
    c_probability[i] = len(temp_df) / len(df)
    for j in range(len(var_type)):
        #categorical features
        if var_type[j] == 0:
            assert(var_class_list[j] is not None)
            _dict = {}
            for k in var_class_list[j]:
                n = len(temp_df[temp_df[j] == k])
                N = len(temp_df)
                #avoid zero probability
                _dict[k] = (n + 1) / (N + len(var_class_list[j]))
            total_conditional_probability_dict[i].append(_dict)
        #numerical features
        else:
            assert(var_class_list[j] is None)
            total_conditional_probability_dict[i].append({
                'mean': temp_df[j].astype(float).mean(), 
                'std': temp_df[j].astype(float).std()
            })

#testing
df_test = pd.read_csv(sys.argv[2], header=None, dtype=str)
df_test = df_test.drop([len(var_type)], axis=1)
df_return = df_test.copy()

# filling missing datas
for i in range(len(var_type)):
    missing_count = len(df_test[i][df_test[i] == ' ?'])
    if missing_count != 0:
        #assign an extra category -1 to missing categorical value 
        if var_type[i] == 0:
            df_test[i] = df_test[i].replace(' ?', '-1')
        #assign mean to missing numerical value
        elif var_type[i] == 1:
            mean = df[i][df[i] != ' ?'].astype(float).mean()
            df_test[i] = df_test[i].replace(' ?', str(mean))
        else:
            raise RuntimeError('var_type is wrong')

result = ['class']

for i in range(1, len(df_test)):
    x = df_test.iloc[i].to_numpy()
    predict = {}
    for j in class_list:
        probability = c_probability[j]
        for val, index in zip(x, range(len(x))):
            #categorical features
            if var_type[index] == 0:
                probability *= total_conditional_probability_dict[j][index][val]
            #numerical features
            else:
                probability *= NormalDist(mu=total_conditional_probability_dict[j][index]['mean'], \
                    sigma=total_conditional_probability_dict[j][index]['std']).pdf(float(val))
        predict[j] = probability
    result.append(max(predict, key=predict.get))

#save result to csv
df_return[len(var_type)] = result
df_return.to_csv(sys.argv[3], header=False, index=False)