import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import pickle
training_images2 = pickle.load(open("X.pickle","rb"))
training_labels= pickle.load(open("y.pickle","rb"))

training_images2 = training_images2/255.0

training_labels=np.array(training_labels)






model = Sequential()


model.add(Flatten(input_shape=(100,75,1))) 
#model.add(Flatten()) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation="softmax"))

                                
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images2, training_labels, epochs=3,validation_split=0.1)





model.save("All Numbers no pool")