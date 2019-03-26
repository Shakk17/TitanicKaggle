import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# Load the train and test data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
full_data = [train, test]

print("Null values in training:\n", train.isnull().sum())
print("Null values in test:\n", test.isnull().sum())

# Drop unnecessary features
train = train.drop(["Name", "Ticket"], axis=1)
test = test.drop(["Name", "Ticket"], axis=1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train["Embarked"] = train["Embarked"].fillna("S")


