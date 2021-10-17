import pandas as pd
import seaborn as sns
from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print

import pyod
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

data = pd.read_csv('arrhythmia.data.txt', sep=",", header=None)

data.drop(data.columns[169:279],axis=1,inplace=True)
data.drop(data.columns[27:168],axis=1,inplace=True)

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

data.columns=["Age","Sex","Height","Weight","QRS duration","P-R","Q-T","T","P","aQRS","aT","aP","aQRST","J","Heart Rate",
              "wQ","wR","wS","wR_","wS_","Number of intrinsic deflections","Existence of ragged R wave","Existence of diphasic derivation of R wave",
             "Existence of ragged P wave","Existence of diphasic derivation of P wave","Existence of ragged T wave","Existence of diphasic derivation of T wave",
             "QRSA","QRSTA"]

data.drop(columns='wS_', inplace=True)

data = data.mask(data == '?', 0)

data["diagnosis"]=data["Existence of ragged R wave"]+data["Existence of ragged P wave"]+data["Existence of ragged T wave"]
+data["Existence of diphasic derivation of P wave"]+data["Existence of diphasic derivation of R wave"]+\
data["Existence of diphasic derivation of T wave"]
data=data.drop(columns=['Existence of ragged R wave',
       'Existence of diphasic derivation of R wave',
       'Existence of ragged P wave',
       'Existence of diphasic derivation of P wave',
       'Existence of ragged T wave',
       'Existence of diphasic derivation of T wave'])
Y_data=data.iloc[:,-1]
X_data=data.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(X_data,Y_data,test_size=0.33, random_state=42)

clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)

# get the prediction label and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_test_pred)

corr=data.corr(method="pearson")
sns.heatmap(corr)

joblib.dump(clf, 'trained_anomaly detection.pkl')
