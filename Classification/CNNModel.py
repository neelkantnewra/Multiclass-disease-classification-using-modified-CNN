'''
@ author neelkant newra
13.04.2022

Model Architecture to be used for multiclass Classification 

All model are set for 4 classes ['Pneumonia','Normal','Covid-19','Tuberculosis']
'''



from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,Activation,MaxPooling2D
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision,Recall

# VGG-13 model
def VGG13(input_shape):
    model = Sequential()
    model.add(Conv2D(64,3,padding='same',activation='relu',input_shape = input_shape,strides = 1))
    model.add(Conv2D(64,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128,3,activation='relu'))
    model.add(Conv2D(128,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256,3,activation='relu'))
    model.add(Conv2D(256,3,activation='relu'))
    model.add(Conv2D(256,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(512,3,activation='relu'))
    model.add(Conv2D(512,3,activation='relu'))
    model.add(Conv2D(512,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    
    model.add(Dense(4,input_dim=input_shape[0],activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy',Precision(),Recall()])
    
    return model
  
# Base CNN model
def CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(128,3,padding='same',activation='relu',input_shape = input_shape,strides = 1))
    model.add(Conv2D(128,5,padding='same',activation='relu',strides = 1))
    model.add(Conv2D(128,7,padding='same',activation='relu',strides = 1))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,input_dim=input_shape[0],activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy',Precision(),Recall()])
    return model
