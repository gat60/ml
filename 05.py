import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

iris = datasets.load_iris()
print(iris.target_names)

x, y = datasets.load_iris(return_x_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

data = pd.DataFrame({
    'sepalength': iris.data[:, 0],
    'sepalwidth': iris.data[:, 1],
    'petallength': iris.data[:, 2],
    'petalwidth': iris.data[:, 3],
    'species': iris.target
})
print(data.head())

df = RandomForestClassifier(n_estimators=100)
df.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("Accuracy of the model:", metrics.accuracy_score(y_test, y_pred))

print(df.predict([[3, 3, 2, 2]]))
print(iris.target_names[clf.predict([[3, 3, 2, 2]])])
