from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
path=os.path.dirname(__file__)
path_df=path+r'/recons_dataset/new_dataset.csv'
df=pd.read_csv(path_df)
df['num']=df['num']
train, test=train_test_split(df, test_size=0.1)
df1=train
df=train.drop('num', axis=1)
X_train=df
Y_train=df1['num']


df1=test
df=test.drop('num', axis=1)
X_test=df
Y_test=df1['num']

# model=Sequential()
# model.add(Dense(10, input_dim=10, activation='relu'))
# model.add(Dense(25,  activation='relu'))
# model.add(Dense(5,activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.externals import joblib
clf=OneVsRestClassifier(estimator=RandomForestClassifier())
from sklearn.externals import joblib
class_weight={0: 1, 1:10, 2:10, 3:10, 4:5}
# model.fit(X_train, Y_train, class_weight=class_weight, epochs=30)
# h=model.predict_classes(X_test)
clf.fit(X_train, Y_train)
h=clf.predict(X_test)
print(accuracy_score(h, Y_test))
# model_json=model.to_json()
# with open(os.path.dirname(__file__)+'/models/model.json','w') as json_file:
#     json_file.write(model_json)
# joblib.dump(scaler, filename=os.path.dirname(__file__)+'scaler_obj.pkl')
# model.save_weights(os.path.dirname(__file__)+"model.h5")

