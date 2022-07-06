import os
import pandas as pd
from IPython.display import display

dir_list = os.listdir("datasets")  # retrieve all files of current category

dataset = {}
for dataset_dir in dir_list:  # foreach csv file
    dataset[dataset_dir] = list()
    file_list = os.listdir("datasets/" + dataset_dir)
    for dataset_file in file_list:
        current_dataset = pd.read_csv("datasets/" + dataset_dir + "/" + dataset_file)  # parse csv
        dataset[dataset_dir].append(current_dataset)  # append this csv to the other
    dataset[dataset_dir] = pd.concat(dataset[dataset_dir], ignore_index=True)

display(dataset['haipeng_li'])