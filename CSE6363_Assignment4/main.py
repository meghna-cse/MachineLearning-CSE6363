import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTree
from random_forest import RandomForest
from adaboost import AdaBoost

# Load the dataset
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Handle missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train['Fare'].fillna(train['Fare'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Feature engineering
train['FamilySize'] = train['SibSp'] + train['Parch']
test['FamilySize'] = test['SibSp'] + test['Parch']
train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Converting categorical variables
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

title_mapping = {title: i for i, title in enumerate(train['Title'].unique())}
train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)

embarked_mapping = {port: i for i, port in enumerate(train['Embarked'].unique())}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

# Drop unnecessary columns
train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Display the first few rows of the processed data
print("Processed Training Data:")
print(train.head())
print("\n" + "#" * 50 + "\n")

# Split the data
X = train.drop('Survived', axis=1).values
y = train['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the models
dt = DecisionTree(max_depth=5, criterion='gini', min_samples_split=2, min_samples_leaf=1)
rf = RandomForest(DecisionTree, num_trees=10, min_features=3)
ab = AdaBoost(DecisionTree, num_learners=50, learning_rate=1)

# Train the models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
ab.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_ab = ab.predict(X_test)

# Evaluate performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_ab = accuracy_score(y_test, y_pred_ab)

results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'AdaBoost'],
    'Accuracy': [accuracy_dt, accuracy_rf, accuracy_ab]
})
print(results)

