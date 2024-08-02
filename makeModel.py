import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from keras.api.models import Sequential
from keras.api.layers import Dense, Input


def show_image(x_train, y_train, amount):
  plt.figure(figsize=(10 ,8))
  for i in np.arange(1,amount):
      plt.subplot(int('33'+str(i)))
      plt.imshow(x_train[i+10].reshape(28, 28), cmap=plt.cm.binary)
      plt.title(y_train[i+10])
      plt.axis('off')
  plt.show()


data = pd.read_csv('data\\train.csv')

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x, y = data.drop(['label'],axis=1).values, data['label']

# print(x.shape[1])

num_train = 38000
x_train,x_test,y_train,y_test = x[:num_train], x[num_train:], y[:num_train], y[num_train:]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Input((x.shape[1],)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, verbose=2)


model.evaluate(x_test, y_test)

model.save('Number Reader.keras')
