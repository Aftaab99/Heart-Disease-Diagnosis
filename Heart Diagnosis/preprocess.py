import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
path=os.path.dirname(__file__)
path1=path+r'/dataset/processed.cleveland.csv'

df=pd.read_csv(path1)

df=df.replace('?', np.nan)

df=df.dropna()


df.columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']


print(df)
df=df.drop(['oldpeak', 'slope','ca'], axis=1)

df.to_csv(path+r'\recons_dataset\new_dataset.csv', index=False)