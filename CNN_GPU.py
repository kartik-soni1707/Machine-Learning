#Building CNN
from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
classifier=Sequential()

classifier.add(Conv2D(64,(3,3),input_shape=(256,256,3),activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(rate=0.4))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=5,
        validation_data=test_set,
        validation_steps=2000)
test_set2 = test_datagen.flow_from_directory(
        'dataset/single_prediction',
        target_size=(256, 256),
        class_mode='binary')
from PIL import Image
from numpy import asarray
import numpy as np
img = Image.open("cat_or_dog_1.jpg")  
img=img.resize((256,256))
img=asarray( img)
img=img/255
img2 = Image.open("cat_or_dog_2.jpg")  
img2=img2.resize((256,256))
img2=asarray( img2)
img2=img/255
img = np.expand_dims(img, axis=0)
img2=np.expand_dims(img2, axis=0)
img3 = Image.open("download.jpg")  
img3=img3.resize((256,256))
img3=asarray( img3)

img3=np.expand_dims(img3, axis=0)
a=classifier.predict(img)
if(a>0.5):
    print("Dog")
else:    
    print("Cat")


          

