import pandas as pd
path = r'enjoysport.csv'
df=pd.read_csv(path)
df.head()

df['Newvalue']=df['humidity']
print('processed data')
df.head()

df.to_csv('exp_data.csv',index=False)
print('data exported')
df1=pd.read_csv('exp_data.csv')
df1.head()

excel_path=r'marks.xlsx'
df=pd.read_excel(excel_path)
df.head()

df['Total']=df.sum(axis=1)
df.head()

df.to_excel('excel_new.xlsx',index=False)
df1=pd.read_excel(excel_new.xlsx)
