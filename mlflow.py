import os
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
import mlflow

final_dataset = pd.read_csv("datasets//final_dataset.csv")
X = final_dataset.iloc[:, 0:7].values
y = final_dataset.iloc[:, 7].values

# oversampling dataset values
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

counter = Counter(y)

# split dataset using 70/30 technique
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=6)

# min-max transformation on dataset
std_scaler = preprocessing.StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)
pd.DataFrame(X_train_scaled)



