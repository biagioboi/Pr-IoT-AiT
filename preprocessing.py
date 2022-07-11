import os
import pandas as pd
import math
from IPython.display import display

features = ['date', 'length', 'dstip', 'dstport', 'highest_layer',
            'delta', 'ack_flag', 'microphone', 'content_type', 'synchronized', 'class']

dir_list = os.listdir("datasets")  # retrieve all files of current category

dataset = {}
for dataset_dir in dir_list:  # foreach csv file
    dataset[dataset_dir] = list()
    file_list = os.listdir("datasets/" + dataset_dir)
    for dataset_file in file_list:
        current_dataset = pd.read_csv("datasets/" + dataset_dir + "/" + dataset_file)  # parse csv
        dataset[dataset_dir].append(current_dataset)  # append this csv to the other
    dataset[dataset_dir] = pd.concat(dataset[dataset_dir], ignore_index=True)


# In order to proceed with preprocessing phase we need to check that null values are only related to content_type
# Change each null value with 0.0, in order to avoid problems
new_values = []
for y in dataset:
    print(y)
    for feature in features:
        if dataset[y].isna().sum()[feature] and feature != "content_type":
            # To implement some mechanism able to notify the user
            raise Exception("Unable to proceed, please fix the dataset" + y + ", problems related to feature " + feature)
    for value in dataset[y]['content_type']:
        if math.isnan(value):
            new_values.append(0.0)
        else:
            new_values.append(value)
    dataset[y].update({'content_type': new_values})