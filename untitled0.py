# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:27:50 2020

@author: wwj
"""
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
#%%

mu, sigma = 0, .1
data_size = (10000,200) #10000,100000,1000000 data
target_size = (10000,2) # 2,10,100類 target

data = np.random.normal(loc=0, scale=.1, size=(10000,200))
target = np.random.randint(2, size=(10000,1))
# print(len(data), len(target))

#%%
train_x = data[:8000]
test_x = data[8000:]
train_y = target[:8000]
test_y = target[8000:]

#%%


model = Sequential()
model.add(Dense(50, input_shape=(200,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(train_x,train_y,batch_size=64, epochs=50, verbose=1, callbacks=early_stopping)