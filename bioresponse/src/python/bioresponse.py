from __future__ import print_function
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Load training set
train = pd.read_csv('../../data/train.csv')

# Create training set
X = np.array(train.values[0::, 1::])
y = np.array(train.values[0::, 0])

# Gradient boosting classifier
clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.2, verbose=False, random_state=241)
clf.fit(X, y)
p = clf.predict(X).astype(int)

accuracy = accuracy_score(y, p)
precision = precision_score(y, p)
recall = recall_score(y, p)
f1 = f1_score(y, p)
roc_auc = roc_auc_score(y, p)

print("accuracy = %.3f" % accuracy)
print("precision = %.3f" % precision)
print("recall = %.3f" % recall)
print("f1 = %.3f" % f1)
print("roc_auc = %.3f" % roc_auc)

# Load testing set
test = pd.read_csv('../../data/test.csv')
ids = range(1, len(test)+1)

# Predict by the model
output = map(lambda x: x[1], clf.predict_proba(test))

f = open("bioresponse.csv", "wb")
writer = csv.writer(f)
writer.writerow(["MoleculeId","PredictedProbability"])
for row in zip(ids, output):
    writer.writerow((row[0], '%.6f' % row[1]))
f.close()
