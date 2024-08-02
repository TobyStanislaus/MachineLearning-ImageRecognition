import random
import numpy as np
import tensorflow as tf
import pickle
import os

from PIL import Image,ImageOps

newTrainingImages=[]
newTrainingLabels=[]

################################
###Inputting my training Data###
################################





for image in os.listdir(photo1):
    
    img=Image.open(photo1+"\\"+image)   
    #image =img.resize((75,100))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis1Photos.append((arr,1))







photo1=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\1"
photo2=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\2"
photo3=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\3"
photo4=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\4"
photo5=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\5"
photo6=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\6"
photo7=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\7"
photo8=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\8"
photo9=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\9"
photo0=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\0"
photoNone=r"C:\Users\toby\OneDrive\Documents\Python\hello\TensorFlow\Tensorflow Number Reader\New\Numbers B+W\None"

lis1Photos=[]
lis2Photos=[]
lis3Photos=[]
lis4Photos=[]
lis5Photos=[]
lis6Photos=[]
lis7Photos=[]
lis8Photos=[]
lis9Photos=[]
lis0Photos=[]
nonePhotos=[]


##############################
####ACTUAL NUMBER IMAGES#####
##############################

for image in os.listdir(photo1):
    
    img=Image.open(photo1+"\\"+image)   
    #image =img.resize((75,100))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis1Photos.append((arr,1))

for image in os.listdir(photo2):
    img=Image.open(photo2+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis2Photos.append((arr,2))

for image in os.listdir(photo3):
    img=Image.open(photo3+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis3Photos.append((arr,3))

for image in os.listdir(photo4):
    img=Image.open(photo4+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis4Photos.append((arr,4))

for image in os.listdir(photo5):
    img=Image.open(photo5+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis5Photos.append((arr,5))

for image in os.listdir(photo6):
    #img=Image.open(photo6+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis6Photos.append((arr,6))

for image in os.listdir(photo7):
    img=Image.open(photo7+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis7Photos.append((arr,7))

for image in os.listdir(photo8):
    img=Image.open(photo8+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis8Photos.append((arr,8))

for image in os.listdir(photo9):
    img=Image.open(photo9+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis9Photos.append((arr,9))

for image in os.listdir(photo0):
    img=Image.open(photo0+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    lis0Photos.append((arr,0))

for image in os.listdir(photoNone):
    img=Image.open(photoNone+"\\"+image)   
    #image =img.resize((30,70))
    #bw = image.convert("1",dither=Image.NONE)
    #bw.show()
    arr=np.asarray(img)
    nonePhotos.append((arr,10))
#########################



coupledList=lis1Photos+lis2Photos+lis3Photos+lis4Photos+lis5Photos+lis6Photos+lis7Photos+lis8Photos+lis9Photos+lis0Photos+nonePhotos

random.shuffle(coupledList)

for c in range(0,len(coupledList)):
    image,label=coupledList[c]
    newTrainingImages=np.append(newTrainingImages,image)
    newTrainingLabels.append(label)

IMAGES_AMOUNT=len(newTrainingLabels)

training_images2=newTrainingImages
training_labels=newTrainingLabels

training_images2=np.array(training_images2)
training_images2=training_images2.reshape(IMAGES_AMOUNT,100,75,1)




pickle_out = open("X.pickle","wb")
pickle.dump(training_images2,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(training_labels,pickle_out)
pickle_out.close()

