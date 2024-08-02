import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.api.datasets import mnist


data = pd.read_csv('data\\train.csv')

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x, x = data.drop(['label'],axis=1).values, data['label']

print(x.shape)
print()




