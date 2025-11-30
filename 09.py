import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

correct = []
incorrect = []

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        correct.append((X_test[i], y_test[i], y_pred[i]))
    else:
        incorrect.append((X_test[i], y_test[i], y_pred[i]))

print("Correct Predictions:")
for sample in correct:
    print(f"Features: {sample[0]}, Actual: {iris.target_names[sample[1]]}, Predicted: {iris.target_names[sample[2]]}")

if incorrect:
    print("\nIncorrect Predictions:")
    for sample in incorrect:
        print(f"Features: {sample[0]}, Actual: {iris.target_names[sample[1]]}, Predicted: {iris.target_names[sample[2]]}")
else:
    print("\nNo Incorrect Predictions!")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
