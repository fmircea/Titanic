import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv", index_col='PassengerId')

data['Age'] = data['Age'].fillna(value=data['Age'].mean())

data ['Sex'] = data['Sex'].replace(['male', 'female'],[1, 0])

data ['Embarked'] = data['Embarked'].replace(['C', 'S', 'Q'],[1, 2, 3])
data['Embarked'] = data['Embarked'].fillna(value=random.choice(data['Embarked']))

features = data[['Age', 'Fare', 'Sex', 'Pclass']].copy()
answers = data['Survived']

model = KNeighborsClassifier(n_neighbors=5)
model.fit(features[:-100], answers[:-100])

test_predictions = model.predict(features[-100:])
print("Test accuracy:", accuracy_score(answers[-100:], test_predictions))