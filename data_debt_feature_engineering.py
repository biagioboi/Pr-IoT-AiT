import os
import pandas as pd
import math
import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt

features = ['date', 'length', 'dstip', 'dstport', 'highest_layer',
            'delta', 'ack_flag', 'microphone', 'content_type', 'synchronized', 'class']

dir_list = os.listdir("datasets")  # retrieve all files of current category

dataset = {}
for dataset_dir in dir_list:  # foreach csv file
    if dataset_dir == "full_dataset":
        continue
    dataset[dataset_dir] = list()
    file_list = os.listdir("datasets/" + dataset_dir)
    for dataset_file in file_list:
        current_dataset = pd.read_csv("datasets/" + dataset_dir + "/" + dataset_file)  # parse csv
        dataset[dataset_dir].append(current_dataset)  # append this csv to the other
    dataset[dataset_dir] = pd.concat(dataset[dataset_dir], ignore_index=True)



# In order to proceed with preprocessing phase we need to check that null values
# are only related to content_type
# Change each null value with 0.0, in order to avoid problems
# Instead, the rows which contains null values are inserted in the message for the data engineer and deleted.

# Data Debt
new_values = []
for y in dataset:
    print(y)
    for feature in features:
        if dataset[y].isna().sum()[feature] and feature != "content_type":
            list_found = np.where(dataset[y][feature].isnull())[0]
            dataset[y] = dataset[y].dropna(axis=0, subset=[feature])
            to_print = ""
            for x in list_found:
                to_print += str(x) + " - "
            print("Warning: " + feature + " retrieved as null, please fix the script, ref. dataset_name:" + y +
                  " - row: " + to_print)
    for value in dataset[y]['content_type']:
        if math.isnan(value):
            new_values.append(0.0)
        else:
            new_values.append(value)
    dataset[y].update({'content_type': new_values})


# Feature Engineering
for y in dataset:
    dataset[y] = dataset[y].drop(columns=['date', 'dstip', 'synchronized'])
    dataset[y] = dataset[y].drop(dataset[y][dataset[y]['class'] == 'ack'].sample(frac=0.95).index)

# store the three dataset individually and then store the full dataset
to_concat = []
for y in dataset:
    dataset[y].to_csv("datasets/full_dataset/" + y + ".csv", index=False)
    to_concat.append(dataset[y])
full_dataset = pd.concat(to_concat, ignore_index=True)
full_dataset = full_dataset.drop(full_dataset[full_dataset['class'] == 'expected'].sample(frac=0.85).index)
full_dataset.to_csv("datasets/full_dataset/full_dataset.csv", index=False)
