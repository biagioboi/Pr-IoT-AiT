import os
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn import preprocessing
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
import mlflow
import mlflow.sklearn

from IPython.display import display
final_dataset = pd.read_csv("datasets/full_dataset/full_dataset.csv")
X = final_dataset.iloc[:, 0:7].values
y = final_dataset.iloc[:, 7].values
display(X)
# oversampling dataset values
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# min-max transformation on dataset
std_scaler = preprocessing.StandardScaler()

kf = KFold(n_splits=5)


def eval_metrics(actual, pred):
    precision = metrics.precision_score(actual, pred, average= 'weighted')
    recall = metrics.recall_score(actual, pred, average='weighted')
    f1_supp= metrics.f1_score(actual, pred, average='weighted')
    return precision, recall, f1_supp

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state = 6)
counter = Counter(y)
print(counter)
with mlflow.start_run():
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_scaled = std_scaler.fit_transform(X_train)
        one_vs_rest_classifier = OneVsRestClassifier(LinearSVC(random_state=6, max_iter=15000))
        one_vs_rest_classifier.fit(X_train_scaled, y_train)
        X_test_scaled = std_scaler.transform(X_test)
        y_predict = one_vs_rest_classifier.predict(X_test_scaled)
        print(classification_report(y_test, y_predict))
        (precis, rec, f1) = eval_metrics(y_test, y_predict)
        mlflow.log_param("OneVsRest", one_vs_rest_classifier)
        mlflow.log_metric("precision", precis)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
