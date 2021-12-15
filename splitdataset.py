from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv('Twitter_Data(Relabeled).csv')
dataset.columns=['clean_text','category']
dataset['category']=pd.to_numeric(dataset['category'],errors='coerce')
dataset=dataset.dropna(subset=['category'])
dataset['category']=dataset['category'].astype(int)
train , test =train_test_split(dataset,train_size=0.7)
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)