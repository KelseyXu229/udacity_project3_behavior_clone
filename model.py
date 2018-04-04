import cv2
import csv
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda,Flatten,Dropout
from keras.layers import Dense,Activation,Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
samples=[]
with open("./data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples,validation_samples=train_test_split(samples,test_size=0.2)
def generator(samples,batch_size=32,correction=0.2):
    num_samples=len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images=[]
            angles=[]
            for batch_sample in batch_samples:
                center_image=cv2.imread(batch_sample[0])
                center_image=cv2.cvtColor(center_image,cv2.COLOR_RGB2YUV)
                left_image=cv2.imread(batch_sample[1])
                left_image=cv2.cvtColor(left_image,cv2.COLOR_RGB2YUV)
                right_image=cv2.imread(batch_sample[2])
                right_image=cv2.cvtColor(right_image,cv2.COLOR_RGB2YUV)
                center_flip=np.fliplr(center_image)
                center_flip=cv2.cvtColor(center_flip,cv2.COLOR_RGB2YUV)
                
                center_angle=np.float(batch_sample[3])
                left_angle=center_angle+correction
                right_angle=center_angle-correction
                images.extend([center_image,left_image,right_image,center_flip])
                angles.extend([center_angle,left_angle,right_angle,-center_angle])
            X_train=np.array(images)
            y_train=np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)
train_generator=generator(train_samples)
valid_generator=generator(validation_samples)

model=Sequential()

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0-0.5))

model.add(Conv2D(filters=16, kernel_size=3, strides=2, activation='relu',input_shape=(160,320,1)))
model.add(Conv2D(filters=16, kernel_size=3, strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.5))


model.add(Conv2D(filters=28, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(filters=28, kernel_size=3, strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1004, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation=None))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, steps_per_epoch=1000, epochs=8,
                    validation_data=valid_generator, validation_steps=100)
model.save("./data/model.h5")