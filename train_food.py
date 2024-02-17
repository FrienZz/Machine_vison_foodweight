from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd


BATCH_SIZE = 50
IMAGE_SIZE = (256,256)


dataframe = pd.read_csv('contest_foodweight/fried_noodles_dataset.csv', delimiter=',', header=0)


datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:1599],
    directory='contest_foodweight/images',
    x_col='filename',
    y_col=['norm_meat','norm_veggie','norm_noodle'],
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')


validation_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[1600:1856],
    directory='contest_foodweight/images',
    x_col='filename',
    y_col= ['norm_meat','norm_veggie','norm_noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')


inputIm = Input(shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3,))
conv1 = Conv2D(64,3,activation='relu')(inputIm)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPool2D()(conv1)
conv2 = Conv2D(128,3,activation='relu')(pool1)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPool2D()(conv2)
conv3 = Conv2D(256,3,activation='relu')(pool2)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPool2D()(conv3)
conv4 = Conv2D(128,3,activation='relu')(pool3)
conv4 = BatchNormalization()(conv4)
pool4 = MaxPool2D()(conv4)
conv5 = Conv2D(64,3,activation='relu')(pool4)
conv5 = BatchNormalization()(conv5)
pool5 = MaxPool2D()(conv5)
flat = Flatten()(pool5)
dense1 = Dense(256,activation='sigmoid')(flat)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(128,activation='sigmoid')(dense1)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(64,activation='sigmoid')(dense1)
dense1 = Dropout(0.5)(dense1)
dense1 = Dense(64,activation='sigmoid')(dense1)
dense1 = Dropout(0.5)(dense1)
predictedW = Dense(3,activation='sigmoid')(dense1)

model = Model(inputs=inputIm, outputs=predictedW)

model.compile(optimizer=Adam(lr = 1e-4), loss='mse', metrics=['mean_absolute_error'])
model.summary()


checkpoint = ModelCheckpoint('foodweight_best.h5', verbose=1, monitor='val_mean_absolute_error',save_best_only=True, mode='min')

#Train Model
model.fit_generator(
    train_generator,
    steps_per_epoch= len(train_generator),
    epochs=90,
    validation_data=validation_generator,
    validation_steps= len(validation_generator),
    callbacks=[checkpoint])
