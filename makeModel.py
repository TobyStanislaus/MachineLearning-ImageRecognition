import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.api.utils import to_categorical
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Input



def show_image(imageData, label, amount):

    plt.figure(figsize=(10, 8))
    for i in np.arange(1, amount):
        plt.subplot(int('33'+str(i)))
        plt.imshow(imageData[i+10].reshape(28, 28), cmap=plt.cm.binary)
        plt.title(label[i+10])
        plt.axis('off')
    plt.show()


def transform_data(path):
    data = pd.read_csv(path)

    x, y = data.drop(['label'], axis=1).values, data['label']

    num_train = 38000
    x_train, x_test, y_train, y_test = x[:num_train], x[num_train:], y[:num_train], y[num_train:]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, x_test, y_train, y_test, (x.shape[1], 1)


def build_model(path):
    x_train, x_test, y_train, y_test, dataShape = transform_data(path)

    model = Sequential()

    model.add(Input(dataShape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30, verbose=2)

    # model.evaluate(x_test, y_test)

    # model.save('Number Reader.keras')


def use_model(dir):
    '''
    Input the directory of images you want to test,
    and it will predict the numbers.
    '''

    number_reader = load_model('Number Reader.keras')

    # initialize the np array which we will add the image data to
    images = np.zeros(shape=(len(os.listdir(dir)), 784))
    i = 0

    # iterate through the directory and append the image arrays to images
    for path in os.listdir(dir):
        arr = convert_im_to_arr(dir, path)
        images[i] = arr
        i += 1

    preds = number_reader.predict(images)
    for pred in preds:
        print(np.argmax(pred))


def convert_im_to_arr(dir, path):
    im = cv2.imread(dir+'\\'+path)
    im = cv2.resize(im, (28, 28))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    arr = im.flatten()
    arr = arr.reshape(1, -1)
    arr = np.array(arr)
    return arr


# print(transform_data('data\\train.csv'))
# build_model('data\\train.csv')

#use_model('data\\test')



x_train, x_test, y_train, y_test, shape = transform_data('data\\train.csv')


number_reader = load_model('Number Reader.keras')
prediction = number_reader.predict(x_train)
i=0
for pred in prediction:
  if i==9:
    break
  print(np.argmax(pred))
  i+=1

show_image(x_test, y_test, 10)
print()