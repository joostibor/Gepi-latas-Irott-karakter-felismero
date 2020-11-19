#Szükséges csomagok importálása
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#Beolvasott kép újraméterezése és forgatása
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

#Fájlok beolvasása
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

#Adatbetöltés
x_train, y_train = train.iloc[:, 1:], train.iloc[:, 0]
x_test, y_test = test.iloc[:, 1:], test.iloc[:, 0]

#Felesleges pandas adatbázisok törlése
del train
del test

#Normalizálás
x_train = np.apply_along_axis(rotate, 1, x_train.values)
x_test = np.apply_along_axis(rotate, 1, x_test.values)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#Train adatok alapján lehetőségek számának beállítása
number_of_classes = y_train.nunique()
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

#Újraméretezés
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size= 0.10, random_state=88)

#Modell létrehozása
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28,28,1)))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation = tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=16, verbose=1, validation_data=(X_val, y_val))

model.save('digit_reader_conv.model')
