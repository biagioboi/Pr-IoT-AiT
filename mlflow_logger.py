import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn import preprocessing
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import mlflow
import mlflow.sklearn

final_dataset = pd.read_csv("datasets/full_dataset/full_dataset.csv")
X = final_dataset.iloc[:, 0:7].values
y = final_dataset.iloc[:, 7].values

# oversampling dataset values
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

counter = Counter(y)

# min-max transformation on dataset
std_scaler = preprocessing.StandardScaler()

kf = KFold(n_splits=10)


def eval_metrics(actual, pred):
    precision = metrics.precision_score(actual, pred, average='weighted')
    recall = metrics.recall_score(actual, pred, average='weighted')
    f1_supp = metrics.f1_score(actual, pred, average='weighted')
    return precision, recall, f1_supp


models = [{"model_name": "KNN",
           "instance": "KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)",
           "params": [{"k": "7"}, {"p": "2"}]
           }]

for model in models:
    with mlflow.start_run():
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_scaled = std_scaler.fit_transform(X_train)
            X_test_scaled = std_scaler.transform(X_test)
            classifier = eval(model['instance'])
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            (precis, rec, f1) = eval_metrics(y_test, y_pred)
            for element in model['params']:
                for key, param in element.items():
                    mlflow.log_param(key, param)
            mlflow.log_metric("precision", precis)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)
