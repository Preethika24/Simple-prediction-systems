# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv("C:\\Users\\Admin\\Downloads\\insurance proj.csv",encoding="latin1")
x = dataset.iloc[:,0:8].values
y = dataset.iloc[:,9].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
x_train.shape
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
p = np.arange(9).reshape(3,3)
q =sc.fit_transform(p)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(units= 8 ,activation="relu" ,init="uniform"))

model.add(Dense(units= 16 ,activation="relu" ,init="uniform"))
model.add(Dense(units= 1 ,activation="relu" ,init="uniform"))

model.compile(optimizer="adam",loss="mean_squared_error",metrics=["mean_squared_error"])
model.fit(x_train,y_train,epochs=100,batch_size=32)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
mod = pickle.load(open('model.pkl','rb'))
print(mod.predict(np.array([[33,1,22.705,0,0,1,65952,0]])))