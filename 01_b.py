import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv('prg1b.csv')
print('original dataset')
print(df)

missing_data=df.isnull().sum()
print('missing data')
print(missing_data)

imputer=SimpleImputer(Strategy='mean')
df['Salary']=Imputer.fit_transform(df[['Salary']])
df['Age']=Imputer.fit_transform(df[['Age']])

label_encoder=LabelEncoder()
df['Gender']=labelencoder.fit_transform(df['Gender'])
print(df)

ct=ColumnTransform(
    transformers=[('One_Hot',OneHotEncoder(),['Department'])],remainder='passthrough'
)
df_encoded=ct.fit_transform(df)
df_encoded=pd.DataFrame(df_encoded,columns=['HR','IT','Finance','Marketing','ID','Name','Age','Gender','Salary','Location'])
print(df_encoded)
