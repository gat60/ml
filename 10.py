import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

iris=datasets.load_iris()
X=pd.DataFrame(iris.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y=pd.DataFrame(iris.target)
y.columns=['Targets']

scaler=preprocessing.StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)
X_scaled_df=pd.DataFrame(X_scaled,columns=X.columns)

gmm=GaussianMixture(n_components=3)
gmm.fit(X_scaled_df)
gmm_labels=gmm.predict(X_scaled_df)

plt.figure(figsize=(14,14))
colormap=np.array(['red','lime','black'])
plt.subplot(2,2,1)
plt.scatter(X['Petal_Length'],X['Petal_Width'],c=colormap[y['Targets']],s=40)
plt.title('Real clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(2,2,2)
plt.scatter(X['Petal_Length'],X['Petal_Width'],c=colormap[gmm_labels],s=40)
plt.title('GMM Clustering (EM Algorithm)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.tight_layout()
plt.show()
