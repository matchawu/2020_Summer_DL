# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:47:14 2020

@author: wwj
"""
#%%
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
size = 10000
classes = 2
data = np.random.normal(loc=0, scale=.1, size=(size,800))
# target = np.random.normal(loc=0, scale=1, size=(size,1)) # >0 1 <0 0
target = np.random.randint(classes, size=(size,1))
#10000,100000,1000000 data
# 2,10,100類 target

train_x = data[:int(size*0.8)]
test_x = data[int(size*0.8):]
train_y = target[:int(size*0.8)]
test_y = target[int(size*0.8):]

train_y = keras.utils.to_categorical(train_y)
test_y = keras.utils.to_categorical(test_y)

#%%
model = Sequential()
model.add(Dense(50, input_shape=(800,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(classes, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))
model.summary()

#%%
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=0)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])#categorical_crossentropy
history = model.fit(train_x,train_y,batch_size=2048, epochs=50, verbose=1, validation_data=[test_x,test_y])

#%%
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model acc')
plt.ylabel('ACC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# model.predict