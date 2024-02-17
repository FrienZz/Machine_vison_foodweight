from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import csv

BATCH_SIZE = 50
IMAGE_SIZE = (256,256)


dataframe = pd.read_csv('contest_foodweight/fried_noodles_dataset.csv', delimiter=',', header=0)

datagen = ImageDataGenerator(rescale=1./255)

test_generator = datagen.flow_from_dataframe(
    dataframe=dataframe.loc[0:299],
    directory='contest_foodweight/images',
    x_col='filename',
    y_col=  ['meat','veggie','noodle'],
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')


#Test Model
model = load_model('foodweight_best.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)


test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('prediction:\n',predict)

#Write predict data to .csv file
with open('result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename','meat_prediction', 'veggie_prediction','noodle_prediction'])
    for idx in range(299):
        writer.writerow([dataframe.loc[idx][['filename']].item(),predict[idx][0]*47,predict[idx][1]*101,predict[idx][2]*268])
file.close()
