import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

plma = pd.read_csv("diabetes.csv")

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

x = plma[feature_cols]
y = plma['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(15,8))
plot_tree(clf, feature_names=feature_cols, class_names=['No Diabetes','Diabetes'], filled=True)
plt.show()

new_sample = pd.DataFrame({
    'pregnant': [2],
    'insulin': [100],
    'bmi': [30],
    'age': [25],
    'glucose': [100],
    'bp': [70],
    'pedigree': [0.5]
})

prediction = clf.predict(new_sample)

if prediction[0] == 1:
    print("The New Sample is predicted to have diabetes (label = 1)")
else:
    print("The New Sample is predicted to NOT have diabetes (label = 0)")
