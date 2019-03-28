import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the train and test data

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Histograms
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

# Drop unnecessary features
drop_features = ['Cabin', 'Ticket']
train = train.drop(drop_features, axis=1)
train = train.drop(['PassengerId'], axis=1)
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

# Creation of new features FamilySize and IsAlone
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Creation of new feature Title, extracted from Name
for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:
    dataset['Title'] = dataset['Title'].\
        replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in full_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Continuous variable bins
for dataset in full_data:
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

# Convert objects to categories
for dataset in full_data:
    dataset['Sex'] = LabelEncoder().fit_transform(dataset['Sex'])
    dataset['Embarked'] = LabelEncoder().fit_transform(dataset['Embarked'])
    dataset['Age'] = LabelEncoder().fit_transform(dataset['AgeBin'])
    dataset['Fare'] = LabelEncoder().fit_transform(dataset['FareBin'])

# Drop unnecessary features
drop_features = ['AgeBin', 'FamilySize', 'FareBin', 'Name', 'Parch', 'SibSp']
train = train.drop(drop_features, axis=1)
test = test.drop(drop_features, axis=1)

# Visualize correlation
colormap = plt.cm.get_cmap()
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

# Splitting of data
train_X = train.drop("Survived", axis=1)
train_y = train["Survived"]
test_X = test.drop("PassengerId", axis=1).copy()

# Model choice

# Logistic Regression, measures importance of the features.
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(train_X, train_y)
acc_log = round(logreg.score(train_X, train_y) * 100, 2)

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
print(coeff_df)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_y)
pred_y = random_forest.predict(test_X)
random_forest.score(train_X, train_y)
acc_random_forest = round(random_forest.score(train_X, train_y) * 100, 2)
print("Random Forest f-measure: " + str(acc_random_forest))

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred_y
    })
