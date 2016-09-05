from __future__ import print_function
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import GradientBoostingClassifier

# Load training set
train = pd.read_csv('../../data/train.csv')

# Create training set
X = np.array(train.values[0::, 1::])
y = np.array(train.values[0::, 0])

# Gradient boosting classifier
clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.2, verbose=False, random_state=241)
clf.fit(X, y)

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
