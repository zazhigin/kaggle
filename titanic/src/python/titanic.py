from __future__ import print_function
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load training set
train = pd.read_csv('../../data/train.csv')

# Encode Sex (male / female) with Gender (0 / 1)
train['Gender'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Fill missing Age with median Age
median_age = train['Age'].dropna().median()
train.loc[ (train.Age.isnull()), 'Age'] = median_age

# Remove columns which don't need
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

# Create training set
X = np.array(train.values[0::, 1::])
y = np.array(train.values[0::, 0])

# Random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Load testing set
test = pd.read_csv('../../data/test.csv')
ids = test['PassengerId'].values

# Encode Sex (male / female) with Gender (0 / 1)
test['Gender'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Fill missing Age with median
median_age = test['Age'].dropna().median()
test.loc[ (test.Age.isnull()), 'Age'] = median_age

# Fill missing Fare by Pclass median
median_fare_pclass_1 = test[ test.Pclass == 1 ]['Fare'].dropna().median()
median_fare_pclass_2 = test[ test.Pclass == 2 ]['Fare'].dropna().median()
median_fare_pclass_3 = test[ test.Pclass == 3 ]['Fare'].dropna().median()
test.loc[ (test.Fare.isnull()) & (test.Pclass == 1 ), 'Fare'] = median_fare_pclass_1
test.loc[ (test.Fare.isnull()) & (test.Pclass == 2 ), 'Fare'] = median_fare_pclass_2
test.loc[ (test.Fare.isnull()) & (test.Pclass == 3 ), 'Fare'] = median_fare_pclass_3

# Remove columns which don't need
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis=1)

# Predict by the model
output = clf.predict(test).astype(int)

f = open("titanic.csv", "wb")
writer = csv.writer(f)
writer.writerow(["PassengerId","Survived"])
writer.writerows(zip(ids, output))
f.close()
