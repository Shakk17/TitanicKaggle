import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# Load the train and test data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Histograms
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

# Drop unnecessary features
drop_features = ['PassengerId', 'Cabin', 'Name', 'Ticket']
train = train.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)
full_data = [train, test]


# Analysis of possible correlations
for x in train:
    if train[x].dtype != 'float64' and x != 'Survived':
        print('Survival Correlation by:', x)
        print(train[[x, 'Survived']].groupby(x, as_index=False).
              mean().sort_values(by='Survived', ascending=False))
        print('-'*10, '\n')


# Completing data
for dataset in full_data:
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

# Creation of new features
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Continuous variable bins
for dataset in full_data:
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

# Convert objects to categories
for dataset in full_data:
    dataset['Sex_Code'] = LabelEncoder().fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = LabelEncoder().fit_transform(dataset['Embarked'])
    dataset['AgeBin_Code'] = LabelEncoder().fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = LabelEncoder().fit_transform(dataset['FareBin'])

# Drop unnecessary features
drop_features = ['Age', 'AgeBin', 'Embarked', 'FamilySize', 'Fare', 'FareBin',
                 'Parch', 'Sex', 'SibSp']
train = train.drop(drop_features, axis=1)
