import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

def transform_image(dir):
  for path in os.listdir(dir):
    im = cv2.imread(dir+'\\'+path)
    im = cv2.resize(im, (28,28))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im = im.flatten()
    df = pd.DataFrame(im)

    df.to_csv('')



def transform_data(path):
  data = pd.read_csv(path)

  x, y = data.drop(['label'],axis=1).values, data['label']

  num_train = 38000
  x_train, x_test, y_train, y_test = x[:num_train], x[num_train:], y[:num_train], y[num_train:]

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  return x_train, x_test, y_train, y_test, (x.shape[1],)


def build_model(path):
  x_train, x_test, y_train, y_test, dataShape = transform_data(path)

  model = Sequential()

  model.add(Input(dataShape))
  model.add(Dense(10,activation='relu'))
  model.add(Dense(100,activation='relu'))
  model.add(Dense(100,activation='relu'))
  model.add(Dense(10,activation='relu'))

  model.add(Dense(10,activation='softmax'))

  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


  model.fit(x_train, y_train, epochs=30, verbose=2)

  # model.evaluate(x_test, y_test)

  # model.save('Number Reader.keras')

# print(transform_data('data\\train.csv'))
transform_image('data\\test')