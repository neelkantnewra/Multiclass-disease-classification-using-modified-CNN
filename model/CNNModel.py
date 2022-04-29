'''
@ author neelkant newra
13.04.2022

Model Architecture to be used for multiclass Classification 

All model are set for 4 classes ['Pneumonia','Normal','Covid-19','Tuberculosis']
'''

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D,Activation,MaxPooling2D,ZeroPadding2D,GlobalAveragePooling2D,DepthwiseConv2D
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision,Recall
from tensorflow.keras.optimizers import Adam



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
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate=1e-3) , metrics = ['accuracy',Precision(),Recall()])
    
    return model

# AlexNet Model

def AlexNet(input_shape):
    model = Sequential()
    model.add(Conv2D(96,11,activation='relu',input_shape = input_shape,strides = 4))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(256,5,activation='relu',padding='valid',strides = 1))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384,3,activation='relu',padding='valid',strides = 1))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384,3,activation='relu',padding='valid',strides = 1))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256,3,activation='relu',padding='valid',strides = 1))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,input_dim=input_shape[0],activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate=1e-3) , metrics = ['accuracy',Precision(),Recall()])
    return model


# MobileNet 

def convBlock(tensor, channels, strides, alpha=1.0, name=''):
    channels = int(channels * alpha)

    x = Conv2D(channels,
               kernel_size=(3, 3),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_act'.format(name))(x)
    return x


def dwSepBlock(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: Depthwise conv followed by Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn1'.format(name))(x)
    x = Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = BatchNormalization(name='{}_bn2'.format(name))(x)
    x = Activation('relu', name='{}_act2'.format(name))(x)
    return x

def MobileNet(input_shape, num_classes, alpha=1.0 ):
    x_in = Input(shape=input_shape)

    x = convBlock(x_in, 32, (2, 2), alpha=alpha, name='initial')

    layers = [
        (64, (1, 1)),
        (128, (2, 2)),
        (128, (1, 1)),
        (256, (2, 2)),
        (256, (1, 1)),
        (512, (2, 2)),
        *[(512, (1, 1)) for _ in range(5)],
        (1024, (2, 2)),
        (1024, (2, 2))
    ]

    for i, (channels, strides) in enumerate(layers):
        x = dwSepBlock(x, channels, strides, alpha=alpha, name='block{}'.format(i))
        
    x = GlobalAveragePooling2D(name='global_avg')(x)
    x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=x_in, outputs=x)
        
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate=1e-3) , metrics = ['accuracy',Precision(),Recall()])
    return model



# Modified DarkCovidNet

def ModifiedDarkCovidNet(input_shape):
    model = Sequential()
    model.add(Conv2D(8,3,activation='relu',input_shape = input_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(16,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(8,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(16,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(32,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(16,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(64,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(128,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    
    model.add(Conv2D(256,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,3,activation=LeakyReLU(),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(3,3))
    model.add(Flatten())
    model.add(Dense(4,input_dim=input_shape[0],activation='softmax'))
    
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate=1e-3) , metrics = ['accuracy',Precision(),Recall(),AUC()])
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
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(learning_rate=1e-3) , metrics = ['accuracy',Precision(),Recall()])
    return model
